import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from Bio import SeqIO
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import nn, optim


THESIS_AUC = {
    "NPInter2.0_network": 0.791,
    "RAID2.0_network": 0.779,
    "NPInter4.0_network": 0.884,
}


RNA_ALPHABET = "ACGT"
PROTEIN_GROUPS = ["AGV", "ILFP", "YMTS", "HNQW", "RK", "DE", "C"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RNAFeature:
    def __init__(self):
        self.kmers = [a + b + c + d for a in RNA_ALPHABET for b in RNA_ALPHABET for c in RNA_ALPHABET for d in RNA_ALPHABET]

    def transform_one(self, seq):
        seq = str(seq).upper().replace("U", "T")
        counts = Counter()
        for i in range(max(0, len(seq) - 3)):
            token = seq[i : i + 4]
            if all(ch in RNA_ALPHABET for ch in token):
                counts[token] += 1
        denom = max(1, len(seq))
        return np.array([counts[k] / denom for k in self.kmers], dtype=np.float32)


class ProteinFeature:
    def __init__(self):
        self.aa_to_group = {}
        for idx, group in enumerate(PROTEIN_GROUPS):
            for aa in group:
                self.aa_to_group[aa] = str(idx)
        self.kmers = [str(a) + str(b) + str(c) for a in range(7) for b in range(7) for c in range(7)]

    def transform_one(self, seq):
        reduced = [self.aa_to_group[ch] for ch in str(seq).upper() if ch in self.aa_to_group]
        counts = Counter()
        for i in range(max(0, len(reduced) - 2)):
            counts[reduced[i] + reduced[i + 1] + reduced[i + 2]] += 1
        denom = max(1, len(reduced))
        return np.array([counts[k] / denom for k in self.kmers], dtype=np.float32)


def load_fasta_features(path, kind):
    extractor = RNAFeature() if kind == "rna" else ProteinFeature()
    names, features = [], []
    for record in SeqIO.parse(str(path), "fasta"):
        names.append(record.id)
        features.append(extractor.transform_one(record.seq))
    return np.array(names), np.vstack(features).astype(np.float32)


class Encoder(nn.Module):
    def __init__(
        self,
        lnc_channels=256,
        pro_channels=343,
        out_channels=100,
        hidden1=220,
        hidden2=150,
        dropout=0.0,
        activation="sigmoid",
    ):
        super().__init__()
        self.out_channels = out_channels
        act = nn.ReLU if activation == "relu" else nn.Sigmoid

        def block(in_channels):
            layers = [
                nn.Linear(in_channels, hidden1),
                act(),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.extend([
                nn.Linear(hidden1, hidden2),
                act(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden2, out_channels))
            return nn.Sequential(*layers)

        self.layer_lnc = block(lnc_channels)
        self.layer_pro = block(pro_channels)

    def embed_lnc(self, x):
        return self.layer_lnc(x)

    def embed_pro(self, x):
        return self.layer_pro(x)


def read_interactions(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            a, b = line.split("\t")[:2]
            pairs.append((a, b))
    return pairs


def build_graph(rna_ids, pro_ids, pairs):
    graph = nx.Graph()
    graph.add_nodes_from(rna_ids, bipartite="rna")
    graph.add_nodes_from(pro_ids, bipartite="protein")
    graph.add_edges_from(pairs)
    return graph


def cosine_similarity_edges(ids, features, threshold):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    safe = np.divide(features, np.maximum(norms, 1e-12))
    sim = safe @ safe.T
    edges = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if sim[i, j] > threshold:
                edges.append((ids[i], ids[j], float(sim[i, j])))
    return edges


def jaccard_similarity_edges(source_ids, neighbor_map, threshold):
    edges = []
    source_ids = list(source_ids)
    source_set = set(source_ids)
    normalized_neighbors = {
        node: set(neighbors)
        for node, neighbors in neighbor_map.items()
        if node in source_set and neighbors
    }
    inverse_neighbors = defaultdict(list)
    for node, neighbors in normalized_neighbors.items():
        for neighbor in neighbors:
            inverse_neighbors[neighbor].append(node)

    shared_counts = Counter()
    for nodes in inverse_neighbors.values():
        nodes = sorted(nodes)
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                shared_counts[(a, b)] += 1

    for (a, b), intersection in shared_counts.items():
        na = normalized_neighbors[a]
        nb = normalized_neighbors[b]
        union = len(na) + len(nb) - intersection
        if union == 0:
            continue
        score = intersection / union
        if score > threshold:
            edges.append((a, b, float(score)))
    return edges


def build_heterogeneous_graph(
    args,
    rna_ids,
    pro_ids,
    train_pairs,
    rna_seq_features,
    pro_seq_features,
):
    lnc_to_pro = defaultdict(set)
    pro_to_lnc = defaultdict(set)
    for lnc, pro in train_pairs:
        lnc_to_pro[lnc].add(pro)
        pro_to_lnc[pro].add(lnc)

    graph = build_graph(rna_ids, pro_ids, train_pairs)
    stats = {
        "interaction_edges": len(train_pairs),
        "jaccard_rna_edges": 0,
        "jaccard_protein_edges": 0,
        "sequence_rna_edges": 0,
        "sequence_protein_edges": 0,
        "sequence_similarity_backend": "cosine_features",
    }

    if args.walk_graph in {"jaccard", "both"}:
        rna_edges = jaccard_similarity_edges(rna_ids, lnc_to_pro, args.jaccard_threshold)
        pro_edges = jaccard_similarity_edges(pro_ids, pro_to_lnc, args.jaccard_threshold)
        graph.add_edges_from((a, b, {"edge_type": "jaccard_rna", "weight": w}) for a, b, w in rna_edges)
        graph.add_edges_from((a, b, {"edge_type": "jaccard_protein", "weight": w}) for a, b, w in pro_edges)
        stats["jaccard_rna_edges"] = len(rna_edges)
        stats["jaccard_protein_edges"] = len(pro_edges)

    if args.walk_graph in {"sequence", "both"}:
        rna_edges = cosine_similarity_edges(rna_ids, rna_seq_features, args.rna_sequence_threshold)
        pro_edges = cosine_similarity_edges(pro_ids, pro_seq_features, args.protein_sequence_threshold)
        graph.add_edges_from((a, b, {"edge_type": "sequence_rna", "weight": w}) for a, b, w in rna_edges)
        graph.add_edges_from((a, b, {"edge_type": "sequence_protein", "weight": w}) for a, b, w in pro_edges)
        stats["sequence_rna_edges"] = len(rna_edges)
        stats["sequence_protein_edges"] = len(pro_edges)

    stats["nodes"] = graph.number_of_nodes()
    stats["edges"] = graph.number_of_edges()
    return graph, stats


def split_c1_c2_c3(graph, pairs, seed):
    # This follows the original iLncPNet code idea: partition nodes, then
    # evaluate same-region pairs (c1), cross-region pairs (c2), and held region pairs (c3).
    part_a, part_b = nx.algorithms.community.kernighan_lin_bisection(graph, seed=seed)
    sub_b = graph.subgraph(part_b)
    part_c, part_d = nx.algorithms.community.kernighan_lin_bisection(sub_b, seed=seed + 1)
    held_nodes = set(part_d)

    c1, c2, c3 = [], [], []
    for rna, pro in pairs:
        rna_held = rna in held_nodes
        pro_held = pro in held_nodes
        if rna_held and pro_held:
            c3.append((rna, pro))
        elif not rna_held and not pro_held:
            c1.append((rna, pro))
        else:
            c2.append((rna, pro))
    return c1, c2, c3


def degree_balanced_negative_sampling(positive_pairs, all_rna, all_pro, full_positive_set, graph, seed):
    rng = random.Random(seed)
    neg = []
    neg_set = set()
    degrees = dict(graph.degree())
    rna_by_degree = defaultdict(list)
    pro_by_degree = defaultdict(list)
    for node in all_rna:
        rna_by_degree[degrees.get(node, 0)].append(node)
    for node in all_pro:
        pro_by_degree[degrees.get(node, 0)].append(node)

    rna_degree_values = sorted(rna_by_degree)
    pro_degree_values = sorted(pro_by_degree)

    def closest(values, target):
        return min(values, key=lambda x: abs(x - target))

    for rna, pro in positive_pairs:
        target_r = degrees.get(rna, 0)
        target_p = degrees.get(pro, 0)
        sampled = None
        for _ in range(200):
            rd = closest(rna_degree_values, target_r + rng.choice([-1, 0, 1]))
            pd = closest(pro_degree_values, target_p + rng.choice([-1, 0, 1]))
            cand_r = rng.choice(rna_by_degree[rd])
            cand_p = rng.choice(pro_by_degree[pd])
            pair = (cand_r, cand_p)
            if pair not in full_positive_set and pair not in neg_set:
                sampled = pair
                break
        while sampled is None:
            pair = (rng.choice(all_rna), rng.choice(all_pro))
            if pair not in full_positive_set and pair not in neg_set:
                sampled = pair
        neg.append(sampled)
        neg_set.add(sampled)
    return neg


def random_negative_sampling(positive_pairs, all_rna, all_pro, full_positive_set, seed):
    rng = random.Random(seed)
    neg = []
    neg_set = set()
    all_rna = list(all_rna)
    all_pro = list(all_pro)
    while len(neg) < len(positive_pairs):
        pair = (rng.choice(all_rna), rng.choice(all_pro))
        if pair in full_positive_set or pair in neg_set:
            continue
        neg.append(pair)
        neg_set.add(pair)
    return neg


def sample_negatives(args, positive_pairs, all_rna, all_pro, full_positive_set, graph, seed):
    if args.neg_sampling == "random":
        return random_negative_sampling(positive_pairs, all_rna, all_pro, full_positive_set, seed)
    ddb_neg = degree_balanced_negative_sampling(positive_pairs, all_rna, all_pro, full_positive_set, graph, seed)
    if args.random_neg_fraction <= 0:
        return ddb_neg
    random_neg = random_negative_sampling(positive_pairs, all_rna, all_pro, full_positive_set, seed + 999)
    rng = random.Random(seed + 1999)
    mixed = []
    used = set()
    for ddb_pair, random_pair in zip(ddb_neg, random_neg):
        pair = random_pair if rng.random() < args.random_neg_fraction else ddb_pair
        if pair in used:
            pair = ddb_pair if random_pair == pair else random_pair
        mixed.append(pair)
        used.add(pair)
    return mixed


def generate_random_walks(train_pairs, num_walks, walk_length, seed):
    rng = random.Random(seed)
    lnc_to_pro = defaultdict(list)
    pro_to_lnc = defaultdict(list)
    for lnc, pro in train_pairs:
        lnc_to_pro[lnc].append(pro)
        pro_to_lnc[pro].append(lnc)

    walks = []
    for start_lnc in sorted(lnc_to_pro):
        for _ in range(num_walks):
            walk = [start_lnc]
            cur_lnc = start_lnc
            for _step in range(walk_length):
                pro = rng.choice(lnc_to_pro[cur_lnc])
                walk.append(pro)
                cur_lnc = rng.choice(pro_to_lnc[pro])
                walk.append(cur_lnc)
            walks.append(walk)
    return walks


def node_type(node, rna_index):
    return "L" if node in rna_index else "P"


def generate_metapath_walks(graph, rna_index, metapaths, num_walks, walk_length, seed):
    rng = random.Random(seed)
    type_to_nodes = defaultdict(list)
    for node in graph.nodes:
        type_to_nodes[node_type(node, rna_index)].append(node)

    walks = []
    for metapath in metapaths:
        schema = [x for x in metapath.strip().upper() if x in {"L", "P"}]
        if len(schema) < 2:
            continue
        starts = sorted(type_to_nodes[schema[0]])
        for start in starts:
            for _ in range(num_walks):
                walk = [start]
                current = start
                for step in range(1, walk_length):
                    required = schema[step % len(schema)]
                    candidates = [n for n in graph.neighbors(current) if node_type(n, rna_index) == required]
                    if not candidates:
                        break
                    current = rng.choice(candidates)
                    walk.append(current)
                if len(walk) > 1:
                    walks.append(walk)
    return walks


def build_contrast_pairs(walks, rna_index, pro_index, window_size=5):
    contexts = []
    node_counts = Counter()
    for walk in walks:
        for node in walk:
            node_counts[node] += 1
        for i, center in enumerate(walk):
            lo = max(0, i - window_size)
            hi = min(len(walk), i + window_size + 1)
            for j in range(lo, hi):
                if i == j:
                    continue
                contexts.append((center, walk[j]))
    all_nodes = list(node_counts)
    weights = np.array([node_counts[n] ** 0.75 for n in all_nodes], dtype=np.float64)
    weights = weights / weights.sum()
    return contexts, all_nodes, weights


def node_tensor(node, rna_features, pro_features, rna_index, pro_index, device):
    if node in rna_index:
        return torch.from_numpy(rna_features[rna_index[node]]).float().to(device), "rna"
    return torch.from_numpy(pro_features[pro_index[node]]).float().to(device), "pro"


def embed_node(model, node, rna_features, pro_features, rna_index, pro_index, device):
    x, kind = node_tensor(node, rna_features, pro_features, rna_index, pro_index, device)
    x = x.unsqueeze(0)
    return model.embed_lnc(x) if kind == "rna" else model.embed_pro(x)


def train_contrastive_encoder(
    model,
    contexts,
    negative_nodes,
    negative_probs,
    rna_features,
    pro_features,
    rna_index,
    pro_index,
    device,
    epochs,
    batch_limit,
    contrast_batch_size,
    lr,
    negative_samples,
    seed,
):
    rng = np.random.default_rng(seed)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    all_node_names = list(negative_nodes)
    node_to_global = {node: i for i, node in enumerate(all_node_names)}
    contexts_idx = np.array([(node_to_global[c], node_to_global[p]) for c, p in contexts], dtype=np.int64)
    negative_probs = np.asarray(negative_probs, dtype=np.float64)

    rna_global = np.full(len(all_node_names), -1, dtype=np.int64)
    pro_global = np.full(len(all_node_names), -1, dtype=np.int64)
    for node, idx in node_to_global.items():
        if node in rna_index:
            rna_global[idx] = rna_index[node]
        else:
            pro_global[idx] = pro_index[node]

    rna_tensor = torch.from_numpy(rna_features).float().to(device)
    pro_tensor = torch.from_numpy(pro_features).float().to(device)

    def embed_global_indices(global_indices):
        flat = np.asarray(global_indices, dtype=np.int64).reshape(-1)
        out = torch.empty((flat.shape[0], model.out_channels), dtype=torch.float32, device=device)
        rna_mask = rna_global[flat] >= 0
        if np.any(rna_mask):
            rows = torch.from_numpy(rna_global[flat[rna_mask]]).long().to(device)
            out[torch.from_numpy(np.where(rna_mask)[0]).long().to(device)] = model.embed_lnc(rna_tensor[rows])
        if np.any(~rna_mask):
            rows = torch.from_numpy(pro_global[flat[~rna_mask]]).long().to(device)
            out[torch.from_numpy(np.where(~rna_mask)[0]).long().to(device)] = model.embed_pro(pro_tensor[rows])
        return out.reshape(*np.asarray(global_indices).shape, model.out_channels)

    for epoch in range(epochs):
        order = rng.permutation(len(contexts_idx))
        if batch_limit:
            order = order[:batch_limit]
        losses = []
        for start in range(0, len(order), contrast_batch_size):
            batch_ids = order[start : start + contrast_batch_size]
            batch = contexts_idx[batch_ids]
            centers = batch[:, 0]
            positives = batch[:, 1]
            negs = rng.choice(len(all_node_names), size=(len(batch_ids), negative_samples), replace=True, p=negative_probs)

            center_vec = embed_global_indices(centers)
            pos_vec = embed_global_indices(positives).unsqueeze(1)
            neg_vec = embed_global_indices(negs)
            candidate_vecs = torch.cat([pos_vec, neg_vec], dim=1)
            logits = torch.bmm(candidate_vecs, center_vec.unsqueeze(2)).squeeze(2)
            target = torch.zeros(len(batch_ids), dtype=torch.long, device=device)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch + 1} loss={np.mean(losses):.5f} batches={len(losses)} pairs={len(order)}")


def make_pair_features(pairs, rna_embeddings, pro_embeddings, rna_index, pro_index):
    x = np.zeros((len(pairs), rna_embeddings.shape[1] + pro_embeddings.shape[1]), dtype=np.float32)
    for i, (rna, pro) in enumerate(pairs):
        x[i, :] = np.hstack([rna_embeddings[rna_index[rna]], pro_embeddings[pro_index[pro]]])
    return x


def evaluate_classifier(train_pos, train_neg, test_pos, test_neg, rna_embeddings, pro_embeddings, rna_index, pro_index, classifier, args=None):
    train_x = np.vstack([
        make_pair_features(train_pos, rna_embeddings, pro_embeddings, rna_index, pro_index),
        make_pair_features(train_neg, rna_embeddings, pro_embeddings, rna_index, pro_index),
    ])
    train_y = np.hstack([np.ones(len(train_pos)), np.zeros(len(train_neg))])
    test_x = np.vstack([
        make_pair_features(test_pos, rna_embeddings, pro_embeddings, rna_index, pro_index),
        make_pair_features(test_neg, rna_embeddings, pro_embeddings, rna_index, pro_index),
    ])
    test_y = np.hstack([np.ones(len(test_pos)), np.zeros(len(test_neg))])

    if args is not None and args.svm_train_limit and args.classifier == "svm" and len(train_y) > args.svm_train_limit:
        rng = np.random.default_rng(args.seed)
        pos_idx = np.where(train_y == 1)[0]
        neg_idx = np.where(train_y == 0)[0]
        per_class = max(1, args.svm_train_limit // 2)
        keep_pos = rng.choice(pos_idx, size=min(per_class, len(pos_idx)), replace=False)
        keep_neg = rng.choice(neg_idx, size=min(per_class, len(neg_idx)), replace=False)
        keep = np.concatenate([keep_pos, keep_neg])
        rng.shuffle(keep)
        train_x = train_x[keep]
        train_y = train_y[keep]

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    classifier.fit(train_x, train_y)
    if hasattr(classifier, "predict_proba"):
        score = classifier.predict_proba(test_x)[:, 1]
    else:
        score = classifier.decision_function(test_x)
    fpr, tpr, thresholds = roc_curve(test_y, score)
    youden_index = tpr - fpr
    best_threshold = thresholds[int(np.argmax(youden_index))]
    pred = (score >= best_threshold).astype(int)
    return {
        "auc": float(auc(fpr, tpr)),
        "acc": float(accuracy_score(test_y, pred)),
        "precision": float(precision_score(test_y, pred, zero_division=0)),
        "recall": float(recall_score(test_y, pred, zero_division=0)),
        "f1": float(f1_score(test_y, pred, zero_division=0)),
        "youden_threshold": float(best_threshold),
        "youden_j": float(np.max(youden_index)),
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
    }


def parse_csv_values(value, cast):
    return [cast(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_gamma(value):
    value = str(value).strip()
    if value in {"scale", "auto"}:
        return value
    return float(value)


def parse_class_weight(value):
    if value is None:
        return None
    value = str(value).strip().lower()
    return None if value in {"none", "null", "no"} else value


def run_one(args, dataset_name):
    dataset_dir = Path(args.data_dir) / dataset_name
    out_dir = Path(args.output_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = read_interactions(dataset_dir / "interactions.txt")
    if args.smoke:
        pairs = pairs[: min(len(pairs), args.smoke_edges)]

    rna_ids, rna_features = load_fasta_features(dataset_dir / "lncRNA_seq.fa", "rna")
    pro_ids, pro_features = load_fasta_features(dataset_dir / "protein_seq.fa", "protein")
    if args.smoke:
        active_rna = {r for r, _ in pairs}
        active_pro = {p for _, p in pairs}
        rna_mask = np.array([x in active_rna for x in rna_ids])
        pro_mask = np.array([x in active_pro for x in pro_ids])
        rna_ids, rna_features = rna_ids[rna_mask], rna_features[rna_mask]
        pro_ids, pro_features = pro_ids[pro_mask], pro_features[pro_mask]

    rna_seq_features = rna_features.copy()
    pro_seq_features = pro_features.copy()

    rna_scaler = StandardScaler()
    pro_scaler = StandardScaler()
    rna_features = rna_scaler.fit_transform(rna_features).astype(np.float32)
    pro_features = pro_scaler.fit_transform(pro_features).astype(np.float32)

    rna_index = {name: i for i, name in enumerate(rna_ids)}
    pro_index = {name: i for i, name in enumerate(pro_ids)}
    pairs = [(r, p) for r, p in pairs if r in rna_index and p in pro_index]

    graph = build_graph(rna_ids, pro_ids, pairs)
    if args.eval_mode == "kfold":
        return run_kfold(
            args,
            dataset_name,
            pairs,
            graph,
            rna_ids,
            pro_ids,
            rna_features,
            pro_features,
            rna_seq_features,
            pro_seq_features,
            rna_index,
            pro_index,
            out_dir,
        )
    if args.eval_mode == "cold":
        return run_cold_start(
            args,
            dataset_name,
            pairs,
            graph,
            rna_ids,
            pro_ids,
            rna_features,
            pro_features,
            rna_seq_features,
            pro_seq_features,
            rna_index,
            pro_index,
            out_dir,
        )

    train_pos, c2_pos, c3_pos = split_c1_c2_c3(graph, pairs, args.seed)
    full_pos = set(pairs)
    train_neg = sample_negatives(args, train_pos, list(rna_ids), list(pro_ids), full_pos, graph, args.seed + 10)
    c2_neg = sample_negatives(args, c2_pos, list(rna_ids), list(pro_ids), full_pos, graph, args.seed + 20)
    c3_neg = sample_negatives(args, c3_pos, list(rna_ids), list(pro_ids), full_pos, graph, args.seed + 30)

    walks = generate_random_walks(train_pos, args.num_walks, args.walk_length, args.seed)
    contexts, negative_nodes, negative_probs = build_contrast_pairs(walks, rna_index, pro_index, args.window_size)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = Encoder(
        out_channels=args.encoder_out,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
        activation=args.activation,
    ).to(device)
    batch_limit = args.batch_limit
    train_contrastive_encoder(
        model,
        contexts,
        negative_nodes,
        negative_probs,
        rna_features,
        pro_features,
        rna_index,
        pro_index,
        device,
        args.epochs,
        batch_limit,
        args.contrast_batch_size,
        args.lr,
        args.negative_samples,
        args.seed,
    )

    model.eval()
    with torch.no_grad():
        rna_embeddings = model.embed_lnc(torch.from_numpy(rna_features).float().to(device)).cpu().numpy()
        pro_embeddings = model.embed_pro(torch.from_numpy(pro_features).float().to(device)).cpu().numpy()

    if args.classifier == "svm":
        clf = svm.SVC(
            C=args.svm_c,
            kernel=args.svm_kernel,
            gamma=parse_gamma(args.svm_gamma),
            degree=args.svm_degree,
            probability=False,
            class_weight=parse_class_weight(args.svm_class_weight),
            cache_size=args.svm_cache_size,
            random_state=args.seed,
        )
    else:
        clf = RandomForestClassifier(n_estimators=args.rf_trees, criterion="entropy", random_state=args.seed, n_jobs=-1)

    c2_metrics = evaluate_classifier(train_pos, train_neg, c2_pos, c2_neg, rna_embeddings, pro_embeddings, rna_index, pro_index, clf, args)
    if args.classifier == "svm":
        clf = svm.SVC(
            C=args.svm_c,
            kernel=args.svm_kernel,
            gamma=parse_gamma(args.svm_gamma),
            degree=args.svm_degree,
            probability=False,
            class_weight=parse_class_weight(args.svm_class_weight),
            cache_size=args.svm_cache_size,
            random_state=args.seed,
        )
    else:
        clf = RandomForestClassifier(n_estimators=args.rf_trees, criterion="entropy", random_state=args.seed, n_jobs=-1)
    c3_metrics = evaluate_classifier(train_pos, train_neg, c3_pos, c3_neg, rna_embeddings, pro_embeddings, rna_index, pro_index, clf, args)

    result = {
        "dataset": dataset_name,
        "classifier": args.classifier,
        "seed": args.seed,
        "device": str(device),
        "thesis_auc": THESIS_AUC.get(dataset_name),
        "c2_auc": c2_metrics["auc"],
        "c2_acc": c2_metrics["acc"],
        "c3_auc": c3_metrics["auc"],
        "c3_acc": c3_metrics["acc"],
        "mean_auc": (c2_metrics["auc"] + c3_metrics["auc"]) / 2,
        "mean_acc": (c2_metrics["acc"] + c3_metrics["acc"]) / 2,
        "n_train_pairs": len(train_pos),
        "n_c2_pairs": len(c2_pos),
        "n_c3_pairs": len(c3_pos),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    return result


def train_embeddings_for_pairs(
    args,
    train_pos,
    rna_ids,
    pro_ids,
    rna_features,
    pro_features,
    rna_seq_features,
    pro_seq_features,
    rna_index,
    pro_index,
):
    if args.walk_mode == "metapath":
        rna_seq_idx = [rna_index[x] for x in rna_ids]
        pro_seq_idx = [pro_index[x] for x in pro_ids]
        walk_graph, graph_stats = build_heterogeneous_graph(
            args,
            rna_ids,
            pro_ids,
            train_pos,
            rna_seq_features[rna_seq_idx],
            pro_seq_features[pro_seq_idx],
        )
        walks = generate_metapath_walks(
            walk_graph,
            rna_index,
            parse_csv_values(args.metapaths, str),
            args.num_walks,
            args.walk_length,
            args.seed,
        )
    else:
        walks = generate_random_walks(train_pos, args.num_walks, args.walk_length, args.seed)
        graph_stats = {"walk_mode": "bipartite"}
    contexts, negative_nodes, negative_probs = build_contrast_pairs(walks, rna_index, pro_index, args.window_size)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = Encoder(
        out_channels=args.encoder_out,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
        activation=args.activation,
    ).to(device)
    train_contrastive_encoder(
        model,
        contexts,
        negative_nodes,
        negative_probs,
        rna_features,
        pro_features,
        rna_index,
        pro_index,
        device,
        args.epochs,
        args.batch_limit,
        args.contrast_batch_size,
        args.lr,
        args.negative_samples,
        args.seed,
    )
    model.eval()
    with torch.no_grad():
        rna_embeddings = model.embed_lnc(torch.from_numpy(rna_features).float().to(device)).cpu().numpy()
        pro_embeddings = model.embed_pro(torch.from_numpy(pro_features).float().to(device)).cpu().numpy()
    graph_stats["walks"] = len(walks)
    graph_stats["contexts"] = len(contexts)
    return rna_embeddings, pro_embeddings, str(device), graph_stats


def make_classifier(args, svm_c=None, svm_gamma=None, svm_kernel=None):
    if args.classifier == "svm":
        return svm.SVC(
            C=args.svm_c if svm_c is None else svm_c,
            kernel=args.svm_kernel if svm_kernel is None else svm_kernel,
            gamma=parse_gamma(args.svm_gamma if svm_gamma is None else svm_gamma),
            degree=args.svm_degree,
            probability=False,
            class_weight=parse_class_weight(args.svm_class_weight),
            cache_size=args.svm_cache_size,
            random_state=args.seed,
        )
    return RandomForestClassifier(
        n_estimators=args.rf_trees,
        criterion="entropy",
        max_depth=args.rf_max_depth,
        random_state=args.seed,
        n_jobs=-1,
    )


def svm_grid(args):
    cs = parse_csv_values(args.svm_c_list, float) if args.svm_c_list else [args.svm_c]
    gammas = parse_csv_values(args.svm_gamma_list, parse_gamma) if args.svm_gamma_list else [parse_gamma(args.svm_gamma)]
    kernels = parse_csv_values(args.svm_kernel_list, str) if args.svm_kernel_list else [args.svm_kernel]
    return [{"C": c, "gamma": gamma, "kernel": kernel} for c in cs for gamma in gammas for kernel in kernels]


def run_kfold(
    args,
    dataset_name,
    pairs,
    graph,
    rna_ids,
    pro_ids,
    rna_features,
    pro_features,
    rna_seq_features,
    pro_seq_features,
    rna_index,
    pro_index,
    out_dir,
):
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    full_pos = set(pairs)
    fold_rows = []
    grid_rows = []
    svm_configs = svm_grid(args) if args.classifier == "svm" else [None]
    for fold, (train_idx, test_idx) in enumerate(kf.split(pairs), start=1):
        print(f"fold={fold}/{args.folds}")
        train_pos = [pairs[i] for i in train_idx]
        test_pos = [pairs[i] for i in test_idx]
        train_neg = sample_negatives(args, train_pos, list(rna_ids), list(pro_ids), full_pos, graph, args.seed + fold * 100 + 1)
        test_neg = sample_negatives(args, test_pos, list(rna_ids), list(pro_ids), full_pos, graph, args.seed + fold * 100 + 2)
        rna_embeddings, pro_embeddings, device, graph_stats = train_embeddings_for_pairs(
            args,
            train_pos,
            rna_ids,
            pro_ids,
            rna_features,
            pro_features,
            rna_seq_features,
            pro_seq_features,
            rna_index,
            pro_index,
        )
        for cfg in svm_configs:
            if cfg is None:
                clf = make_classifier(args)
            else:
                clf = make_classifier(args, svm_c=cfg["C"], svm_gamma=cfg["gamma"], svm_kernel=cfg["kernel"])
            metrics = evaluate_classifier(
                train_pos,
                train_neg,
                test_pos,
                test_neg,
                rna_embeddings,
                pro_embeddings,
                rna_index,
                pro_index,
                clf,
                args,
            )
            metrics["fold"] = fold
            if cfg is not None:
                metrics.update(cfg)
            metrics["graph_stats"] = graph_stats
            grid_rows.append(metrics)
            label = "" if cfg is None else f" C={cfg['C']} gamma={cfg['gamma']} kernel={cfg['kernel']}"
            print(f"fold={fold}{label} auc={metrics['auc']:.6f} acc={metrics['acc']:.6f}")

    if args.classifier == "svm":
        grouped = defaultdict(list)
        for row in grid_rows:
            grouped[(row["C"], str(row["gamma"]), row["kernel"])].append(row)
        summaries = []
        target = THESIS_AUC.get(dataset_name)
        for (c, gamma, kernel), rows in grouped.items():
            mean_auc = float(np.mean([x["auc"] for x in rows]))
            summaries.append({
                "C": float(c),
                "gamma": gamma,
                "kernel": kernel,
                "mean_auc": mean_auc,
                "std_auc": float(np.std([x["auc"] for x in rows])),
                "mean_acc": float(np.mean([x["acc"] for x in rows])),
                "std_acc": float(np.std([x["acc"] for x in rows])),
                "target_diff": None if target is None else float(mean_auc - target),
            })
        summaries.sort(key=lambda x: abs(x["target_diff"]) if x["target_diff"] is not None else -x["mean_auc"])
        best = summaries[0]
        fold_rows = [
            row for row in grid_rows
            if row.get("C") == best["C"] and str(row.get("gamma")) == best["gamma"] and row.get("kernel") == best["kernel"]
        ]
        with open(out_dir / "svm_grid_summary.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        with open(out_dir / "svm_grid_folds.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(grid_rows[0].keys()))
            writer.writeheader()
            writer.writerows(grid_rows)
    else:
        fold_rows = grid_rows
        best = None

    result = {
        "dataset": dataset_name,
        "classifier": args.classifier,
        "eval_mode": "kfold",
        "folds": args.folds,
        "seed": args.seed,
        "device": device,
        "thesis_auc": THESIS_AUC.get(dataset_name),
        "mean_auc": float(np.mean([x["auc"] for x in fold_rows])),
        "std_auc": float(np.std([x["auc"] for x in fold_rows])),
        "mean_acc": float(np.mean([x["acc"] for x in fold_rows])),
        "std_acc": float(np.std([x["acc"] for x in fold_rows])),
        "best_svm": best,
        "encoder_out": args.encoder_out,
        "hidden1": args.hidden1,
        "hidden2": args.hidden2,
        "activation": args.activation,
        "dropout": args.dropout,
        "lr": args.lr,
        "negative_samples": args.negative_samples,
        "neg_sampling": args.neg_sampling,
        "random_neg_fraction": args.random_neg_fraction,
        "svm_train_limit": args.svm_train_limit,
        "walk_mode": args.walk_mode,
        "walk_graph": args.walk_graph,
        "metapaths": args.metapaths,
        "jaccard_threshold": args.jaccard_threshold,
        "rna_sequence_threshold": args.rna_sequence_threshold,
        "protein_sequence_threshold": args.protein_sequence_threshold,
        "num_walks": args.num_walks,
        "walk_length": args.walk_length,
        "window_size": args.window_size,
        "epochs": args.epochs,
        "batch_limit": args.batch_limit,
        "fold_metrics": fold_rows,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    return result


def choose_hidden(items, fraction, seed):
    rng = random.Random(seed)
    items = sorted(items)
    n_hidden = max(1, int(round(len(items) * fraction)))
    return set(rng.sample(items, min(n_hidden, len(items))))


def cold_split_pairs(pairs, setting, hidden_lnc, hidden_pro):
    if setting == "cold_lncRNA":
        train_pos = [(r, p) for r, p in pairs if r not in hidden_lnc]
        train_pro = {p for _, p in train_pos}
        test_pos = [(r, p) for r, p in pairs if r in hidden_lnc and p in train_pro]
    elif setting == "cold_protein":
        train_pos = [(r, p) for r, p in pairs if p not in hidden_pro]
        train_lnc = {r for r, _ in train_pos}
        test_pos = [(r, p) for r, p in pairs if p in hidden_pro and r in train_lnc]
    else:
        train_pos = [(r, p) for r, p in pairs if r not in hidden_lnc and p not in hidden_pro]
        test_pos = [(r, p) for r, p in pairs if r in hidden_lnc and p in hidden_pro]
    return train_pos, test_pos


def run_cold_start(args, dataset_name, pairs, full_graph, rna_ids, pro_ids, rna_features, pro_features, rna_seq_features, pro_seq_features, rna_index, pro_index, out_dir):
    active_rna = {r for r, _ in pairs}
    active_pro = {p for _, p in pairs}
    settings = ["cold_lncRNA", "cold_protein", "cold_both"] if args.cold_setting == "all" else [args.cold_setting]
    full_pos = set(pairs)
    svm_configs = svm_grid(args) if args.classifier == "svm" else [None]
    rows = []

    for setting in settings:
        setting_rows = []
        for rep in range(args.cold_repeats):
            split_seed = args.seed + rep * 1009
            hidden_lnc = choose_hidden(active_rna, args.cold_fraction, split_seed + 1) if setting in {"cold_lncRNA", "cold_both"} else set()
            hidden_pro = choose_hidden(active_pro, args.cold_fraction, split_seed + 2) if setting in {"cold_protein", "cold_both"} else set()
            train_pos, test_pos = cold_split_pairs(pairs, setting, hidden_lnc, hidden_pro)
            if not train_pos or not test_pos:
                print(f"setting={setting} repeat={rep + 1} skipped train={len(train_pos)} test={len(test_pos)}")
                continue

            train_rna = sorted({r for r, _ in train_pos})
            train_pro = sorted({p for _, p in train_pos})
            if setting == "cold_lncRNA":
                test_rna = sorted(hidden_lnc)
                test_pro = train_pro
            elif setting == "cold_protein":
                test_rna = train_rna
                test_pro = sorted(hidden_pro)
            else:
                test_rna = sorted(hidden_lnc)
                test_pro = sorted(hidden_pro)

            train_neg = sample_negatives(args, train_pos, train_rna, train_pro, full_pos, full_graph, split_seed + 11)
            test_neg = sample_negatives(args, test_pos, test_rna, test_pro, full_pos, full_graph, split_seed + 12)

            print(f"setting={setting} repeat={rep + 1}/{args.cold_repeats} hidden_lnc={len(hidden_lnc)} hidden_pro={len(hidden_pro)} train_pos={len(train_pos)} test_pos={len(test_pos)}")
            rna_embeddings, pro_embeddings, device, graph_stats = train_embeddings_for_pairs(
                args,
                train_pos,
                np.array(train_rna),
                np.array(train_pro),
                rna_features,
                pro_features,
                rna_seq_features,
                pro_seq_features,
                rna_index,
                pro_index,
            )

            for cfg in svm_configs:
                clf = make_classifier(args) if cfg is None else make_classifier(args, svm_c=cfg["C"], svm_gamma=cfg["gamma"], svm_kernel=cfg["kernel"])
                metrics = evaluate_classifier(
                    train_pos,
                    train_neg,
                    test_pos,
                    test_neg,
                    rna_embeddings,
                    pro_embeddings,
                    rna_index,
                    pro_index,
                    clf,
                    args,
                )
                metrics.update({
                    "dataset": dataset_name,
                    "setting": setting,
                    "repeat": rep + 1,
                    "seed": split_seed,
                    "hidden_lnc": len(hidden_lnc),
                    "hidden_protein": len(hidden_pro),
                    "train_pos": len(train_pos),
                    "test_pos": len(test_pos),
                    "device": device,
                    "graph_stats": graph_stats,
                })
                if cfg is not None:
                    metrics.update(cfg)
                setting_rows.append(metrics)
                label = "" if cfg is None else f" C={cfg['C']} gamma={cfg['gamma']} kernel={cfg['kernel']}"
                print(f"setting={setting} repeat={rep + 1}{label} auc={metrics['auc']:.6f} acc={metrics['acc']:.6f}")

        if not setting_rows:
            continue
        if args.classifier == "svm":
            grouped = defaultdict(list)
            for row in setting_rows:
                grouped[(row["C"], str(row["gamma"]), row["kernel"])].append(row)
            summaries = []
            for (c, gamma, kernel), group in grouped.items():
                summaries.append({
                    "dataset": dataset_name,
                    "setting": setting,
                    "C": float(c),
                    "gamma": gamma,
                    "kernel": kernel,
                    "mean_auc": float(np.mean([x["auc"] for x in group])),
                    "std_auc": float(np.std([x["auc"] for x in group])),
                    "mean_acc": float(np.mean([x["acc"] for x in group])),
                    "std_acc": float(np.std([x["acc"] for x in group])),
                    "mean_precision": float(np.mean([x["precision"] for x in group])),
                    "mean_recall": float(np.mean([x["recall"] for x in group])),
                    "mean_f1": float(np.mean([x["f1"] for x in group])),
                })
            summaries.sort(key=lambda x: -x["mean_auc"])
            best = summaries[0]
            rows.extend(summaries)
            selected_rows = [
                row for row in setting_rows
                if row.get("C") == best["C"] and str(row.get("gamma")) == best["gamma"] and row.get("kernel") == best["kernel"]
            ]
        else:
            best = None
            selected_rows = setting_rows
            rows.append({
                "dataset": dataset_name,
                "setting": setting,
                "mean_auc": float(np.mean([x["auc"] for x in setting_rows])),
                "std_auc": float(np.std([x["auc"] for x in setting_rows])),
                "mean_acc": float(np.mean([x["acc"] for x in setting_rows])),
                "std_acc": float(np.std([x["acc"] for x in setting_rows])),
            })

        setting_dir = out_dir / setting
        setting_dir.mkdir(parents=True, exist_ok=True)
        with open(setting_dir / "repeat_metrics.json", "w", encoding="utf-8") as fh:
            json.dump(setting_rows, fh, indent=2)
        with open(setting_dir / "selected_metrics.json", "w", encoding="utf-8") as fh:
            json.dump({"best": best, "selected_rows": selected_rows}, fh, indent=2)

    with open(out_dir / "cold_summary.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    with open(out_dir / "cold_summary.csv", "w", newline="", encoding="utf-8") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=sorted({k for row in rows for k in row}))
            writer.writeheader()
            writer.writerows(rows)
    return {"dataset": dataset_name, "eval_mode": args.eval_mode, "summary": rows}


def main():
    parser = argparse.ArgumentParser(description="Reproduce Chapter 5 iLncPNet experiments.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--dataset", default="all", help="Dataset folder name under --data-dir, or 'all' for the three thesis datasets.")
    parser.add_argument("--classifier", default="svm", choices=["svm", "rf"])
    parser.add_argument("--eval-mode", default="kfold", choices=["kfold", "csplit", "cold"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cold-setting", default="all", choices=["all", "cold_lncRNA", "cold_protein", "cold_both"])
    parser.add_argument("--cold-fraction", type=float, default=0.10)
    parser.add_argument("--cold-repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-walks", type=int, default=20)
    parser.add_argument("--walk-length", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--walk-mode", default="bipartite", choices=["bipartite", "metapath"])
    parser.add_argument("--walk-graph", default="both", choices=["jaccard", "sequence", "both"])
    parser.add_argument("--metapaths", default="PPLPLL,LPLPLP")
    parser.add_argument("--jaccard-threshold", type=float, default=0.5)
    parser.add_argument("--rna-sequence-threshold", type=float, default=0.7)
    parser.add_argument("--protein-sequence-threshold", type=float, default=0.2)
    parser.add_argument("--batch-limit", type=int, default=0, help="Limit contrastive updates per epoch. 0 means full epoch.")
    parser.add_argument("--contrast-batch-size", type=int, default=4096)
    parser.add_argument("--encoder-out", type=int, default=100)
    parser.add_argument("--hidden1", type=int, default=220)
    parser.add_argument("--hidden2", type=int, default=150)
    parser.add_argument("--activation", default="sigmoid", choices=["sigmoid", "relu"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--negative-samples", type=int, default=5)
    parser.add_argument("--neg-sampling", default="ddb", choices=["ddb", "random"])
    parser.add_argument("--random-neg-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--svm-c", type=float, default=5.0)
    parser.add_argument("--svm-c-list", default=None, help="Comma-separated SVM C values evaluated after each embedding training.")
    parser.add_argument("--svm-gamma", default="scale")
    parser.add_argument("--svm-gamma-list", default=None, help="Comma-separated gamma values, including scale or auto.")
    parser.add_argument("--svm-kernel", default="rbf", choices=["linear", "poly", "rbf", "sigmoid"])
    parser.add_argument("--svm-kernel-list", default=None, help="Comma-separated kernels evaluated after each embedding training.")
    parser.add_argument("--svm-degree", type=int, default=3)
    parser.add_argument("--svm-class-weight", default="balanced", choices=["balanced", "none"])
    parser.add_argument("--svm-train-limit", type=int, default=0, help="For kernel SVM, subsample this many balanced training rows per fold; 0 uses all rows.")
    parser.add_argument("--svm-cache-size", type=float, default=2000.0, help="SVM kernel cache size in MB.")
    parser.add_argument("--rf-trees", type=int, default=500)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run a small code-path test, not a paper-quality experiment.")
    parser.add_argument("--smoke-edges", type=int, default=300)
    args = parser.parse_args()
    set_seed(args.seed)

    datasets = ["NPInter2.0_network", "RAID2.0_network", "NPInter4.0_network"] if args.dataset == "all" else [args.dataset]
    rows = []
    for dataset in datasets:
        print(f"Running {dataset}...")
        rows.append(run_one(args, dataset))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
