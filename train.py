import bisect
import glob
import os
from tqdm import tqdm
import numpy as np
import pyarrow.parquet as pq

from config import CONDITION_FEATURE_DIM, N_POINTS, PAD_TOKEN_ID, SEQ_LEN
from model import train


VOCAB = {
    "<PAD>": 0,
    "<SOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3,
    "*": 4,
    "+": 5,
    "-": 6,
    "/": 7,
    "0": 8,
    "1": 9,
    "2": 10,
    "3": 11,
    "4": 12,
    "5": 13,
    "6": 14,
    "<constant>": 15,
    "cos": 16,
    "exp": 17,
    "log": 18,
    "pow2": 19,
    "pow3": 20,
    "pow4": 21,
    "pow5": 22,
    "pow6": 23,
    "sin": 24,
    "x1": 25,
    "x2": 26
}


def process_equation_to_tokens(equation):
    if isinstance(equation, np.ndarray):
        if np.issubdtype(equation.dtype, np.integer):
            return equation.astype(np.int64)
        equation = equation.tolist()

    if isinstance(equation, str):
        equation = equation.split()

    if isinstance(equation, (list, tuple)):
        token_ids = []
        for token in equation:
            if isinstance(token, (int, np.integer)):
                token_ids.append(int(token))
                continue

            if isinstance(token, str):
                token_ids.append(VOCAB.get(token, VOCAB["<UNK>"]))
        assert VOCAB["<UNK>"] not in token_ids, f"Warning: Found unknown token(s) in equation: {equation}"

        return np.array(token_ids, dtype=np.int64)

    raise ValueError(f"Unsupported equation format: {type(equation)}")


def pad_or_truncate_tokens(token_ids, target_length=SEQ_LEN, pad_token_id=PAD_TOKEN_ID):
    assert len(token_ids) < target_length, f"Token sequence too long: {len(token_ids)} tokens (max {target_length})"
    if len(token_ids) < target_length:
        padding = np.full(target_length - len(token_ids), pad_token_id, dtype=np.int64)
        return np.concatenate([token_ids, padding])
    return token_ids


def process_coordinates(X, Y, n_points=N_POINTS):
    X = np.vstack(X)
    Y = np.asarray(Y, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if Y.ndim > 1:
        Y = Y.flatten()

    if X.shape[0] != Y.shape[0]:
        print(f"Warning: Mismatched number of points between X and Y. X has {X.shape[0]} points, Y has {Y.shape[0]} points.")
        raise ValueError(f"X and Y must have the same number of points. Got X: {X.shape[0]}, Y: {Y.shape[0]}")

    current_points = X.shape[0]
    if current_points == 0:
        print("Warning: Empty coordinate arrays. Returning zero-padded array.")
        raise ValueError("Empty coordinate arrays")

    if current_points > n_points:
        indices = np.random.choice(current_points, n_points, replace=False)
        indices = np.sort(indices)
        X = X[indices]
        Y = Y[indices]
    elif current_points < n_points:
        pad_size = n_points - current_points
        X_pad = np.repeat(X[-1:], pad_size, axis=0)
        Y_pad = np.repeat(Y[-1:], pad_size, axis=0)
        X = np.concatenate([X, X_pad], axis=0)
        Y = np.concatenate([Y, Y_pad], axis=0)
    assert np.concatenate([X, Y.reshape(-1, 1)], axis=1).shape == (n_points, X.shape[1] + 1), f"Final coordinate shape mismatch: expected {(n_points, X.shape[1] + 1)}, got {np.concatenate([X, Y.reshape(-1, 1)], axis=1).shape}"
    return np.concatenate([X, Y.reshape(-1, 1)], axis=1)


def preprocess_record(item, n_points=N_POINTS, seq_len=SEQ_LEN):
    try:
        equation = item.get("skeleton")
        if equation is None:
            return None

        token_ids = process_equation_to_tokens(equation)
        if token_ids.size == 0:
            token_ids = np.array([PAD_TOKEN_ID], dtype=np.int64)
    except Exception:
        return None

    try:
        X = item["X"]
        Y = item["Y"]
        x_y_combined = process_coordinates(X, Y, n_points)
        return {"token_ids": token_ids, "X_Y_combined": x_y_combined}
    except Exception:
        return None


class LazyLoadingDataset:
    def __init__(self, file_pattern, max_files=None):
        self.file_paths = sorted(glob.glob(file_pattern))
        if not self.file_paths:
            raise FileNotFoundError(f"No parquet files found matching pattern: {file_pattern}")

        if max_files is not None:
            self.file_paths = self.file_paths[:max_files]

        self._files = []
        self._cumulative_rows = []
        total_rows = 0

        print(f"Indexing {len(self.file_paths)} parquet file(s) using metadata...")
        for path in self.file_paths:
            parquet_file = pq.ParquetFile(path)
            metadata = parquet_file.metadata
            num_rows = metadata.num_rows
            row_group_rows = [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]
            row_group_ends = np.cumsum(row_group_rows).tolist()

            self._files.append(
                {
                    "path": path,
                    "parquet": parquet_file,
                    "num_rows": num_rows,
                    "row_group_ends": row_group_ends,
                }
            )
            total_rows += num_rows
            self._cumulative_rows.append(total_rows)

        self._cached_row_group_key = None
        self._cached_row_group_table = None
        print(f"Found {total_rows} total records")

    def __len__(self):
        return self._cumulative_rows[-1] if self._cumulative_rows else 0

    def _file_for_index(self, idx):
        file_idx = bisect.bisect_right(self._cumulative_rows, idx)
        previous_total = self._cumulative_rows[file_idx - 1] if file_idx > 0 else 0
        local_row = idx - previous_total
        return file_idx, local_row

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of bounds: {idx}")

        file_idx, local_row = self._file_for_index(idx)
        file_info = self._files[file_idx]

        row_group_idx = bisect.bisect_right(file_info["row_group_ends"], local_row)
        row_group_start = file_info["row_group_ends"][row_group_idx - 1] if row_group_idx > 0 else 0
        row_in_group = local_row - row_group_start

        cache_key = (file_idx, row_group_idx)
        if self._cached_row_group_key != cache_key:
            parquet_file = file_info["parquet"]
            table = parquet_file.read_row_group(row_group_idx, columns=["skeleton", "X", "Y"])
            self._cached_row_group_table = table
            self._cached_row_group_key = cache_key

        assert self._cached_row_group_table is not None, "Row group data should be cached at this point"
        row = self._cached_row_group_table.slice(row_in_group, 1)
        return {
            "skeleton": row.column("skeleton")[0].as_py(),
            "X": row.column("X")[0].as_py(),
            "Y": row.column("Y")[0].as_py(),
        }


class PreprocessedLazyDataset:
    def __init__(self, lazy_dataset):
        self.lazy_dataset = lazy_dataset
        self._dummy_condition = np.zeros((N_POINTS, CONDITION_FEATURE_DIM), dtype=np.float32)

    def __len__(self):
        return len(self.lazy_dataset)

    def __getitem__(self, idx):
        raw_item = self.lazy_dataset[idx]
        processed_item = preprocess_record(raw_item)

        if processed_item is None:
            return {
                "token_ids": np.array([PAD_TOKEN_ID], dtype=np.int64),
                "X_Y_combined": self._dummy_condition.copy(),
            }

        return processed_item


def compute_normalization_stats(dataset, sample_size=None):
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; cannot compute normalization statistics")

    if sample_size is not None and sample_size > 0:
        sample_size = min(int(sample_size), len(dataset))
        rng = np.random.default_rng(42)
        sampled_indices = np.sort(rng.choice(len(dataset), size=sample_size, replace=False))
        indices_to_process = sampled_indices.tolist()
        print(f"Computing normalization stats over sample ({sample_size}/{len(dataset)} records)...")
    else:
        indices_to_process = list(range(len(dataset)))
        print(f"Computing normalization stats over full dataset ({len(dataset)} records)...")

    x_sum = None
    x_sum_sq = None
    y_sum = 0.0
    y_sum_sq = 0.0
    total_points = 0
    valid_records = 0

    for processed_count, idx in enumerate(tqdm(indices_to_process), start=1):
        if processed_count % 1000 == 0:
            print(f"  Processed {processed_count}/{len(indices_to_process)} records")


        processed_item = preprocess_record(dataset[idx])
        if processed_item is None:
            raise ValueError(f"Failed to preprocess record at index {idx}. This should not happen since we're iterating over the original dataset directly.")

        coords = np.asarray(processed_item["X_Y_combined"], dtype=np.float32)
        coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)

        x_features = coords[:, :-1].astype(np.float64, copy=False)
        y_feature = coords[:, -1].astype(np.float64, copy=False)

        if x_sum is None:
            x_sum = np.zeros(x_features.shape[1], dtype=np.float64)
            x_sum_sq = np.zeros(x_features.shape[1], dtype=np.float64)
        elif x_features.shape[1] != x_sum.shape[0]:
            raise ValueError(
                f"Inconsistent X feature dimension at idx={idx}. "
                f"Expected {x_sum.shape[0]}, got {x_features.shape[1]}"
            )

        x_sum += np.sum(x_features, axis=0)
        x_sum_sq += np.sum(np.square(x_features), axis=0)
        y_sum += float(np.sum(y_feature))
        y_sum_sq += float(np.sum(np.square(y_feature)))

        total_points += x_features.shape[0]
        valid_records += 1

    if valid_records == 0 or total_points == 0:
        raise ValueError(f"No valid records found while computing statistics. {valid_records} valid records, {total_points} total points.")

    x_means = x_sum / total_points
    x_vars = x_sum_sq / total_points - np.square(x_means)
    x_vars = np.maximum(x_vars, 0.0)
    x_stds = np.sqrt(x_vars)

    y_mean = y_sum / total_points
    y_var = y_sum_sq / total_points - (y_mean ** 2)
    y_var = max(y_var, 0.0)
    y_std = float(np.sqrt(y_var))

    x_means = x_means.astype(np.float32)
    x_stds = np.where(x_stds > 1e-6, x_stds, 1.0).astype(np.float32)
    y_std = y_std if y_std > 1e-6 else 1.0

    print("Normalization Stats:")
    for dim in range(len(x_means)):
        print(f"  X dim {dim + 1}: Mean={x_means[dim]:.3f}, Std={x_stds[dim]:.3f}")
    print(f"  Y feature: Mean={y_mean:.3f}, Std={y_std:.3f}")

    return x_means, x_stds, float(y_mean), float(y_std)


def load_and_preprocess_dataset(train_pattern, val_pattern=None, max_train_files=None, max_val_files=None):
    print("=" * 60)
    print("Loading training data (lazy)...")
    lazy_train_dataset = LazyLoadingDataset(train_pattern, max_train_files)
    train_dataset = PreprocessedLazyDataset(lazy_train_dataset)

    val_dataset = None
    if val_pattern:
        print("\nLoading validation data (lazy)...")
        try:
            lazy_val_dataset = LazyLoadingDataset(val_pattern, max_val_files)
            val_dataset = PreprocessedLazyDataset(lazy_val_dataset)
        except FileNotFoundError as err:
            print(f"Warning: {err}")
            print("Continuing without validation data...")

    print("\n" + "=" * 60)
    print("Dataset sizes:")
    print(f"  Training: {len(train_dataset)} samples")
    if val_dataset is not None:
        print(f"  Validation: {len(val_dataset)} samples")
    print("=" * 60)

    return train_dataset, val_dataset


if __name__ == "__main__":
    train_pattern = os.getenv("TRAIN_PATTERN", "data/preprocessed_parquet/Train/*.parquet")
    val_pattern = os.getenv("VAL_PATTERN", "data/preprocessed_parquet/Val/*.parquet")

    max_train_files_env = os.getenv("MAX_TRAIN_FILES")
    max_val_files_env = os.getenv("MAX_VAL_FILES")
    norm_sample_size_env = os.getenv("NORM_SAMPLE_SIZE")
    max_train_files = int(max_train_files_env) if max_train_files_env else None
    max_val_files = int(max_val_files_env) if max_val_files_env else None
    norm_sample_size = int(norm_sample_size_env) if norm_sample_size_env else 5000

    train_dataset, val_dataset = load_and_preprocess_dataset(
        train_pattern=train_pattern,
        val_pattern=val_pattern,
        max_train_files=max_train_files,
        max_val_files=max_val_files,
    )

    x_means, x_stds, y_mean, y_std = compute_normalization_stats(
        train_dataset.lazy_dataset,
        sample_size=norm_sample_size,
    )

    if len(train_dataset) > 0:
        print("\nStarting training...")
        train(
            train_data=train_dataset,
            test_data=val_dataset if val_dataset is not None else [],
            x_means=x_means,
            x_stds=x_stds,
            y_mean=y_mean,
            y_std=y_std,
        )
        print("\nTraining complete!")
    else:
        print("\nError: No training data loaded. Please check your data paths.")