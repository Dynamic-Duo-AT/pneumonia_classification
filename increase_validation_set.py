from pathlib import Path
import random
import shutil
import typer

"""Script to move a fraction of training data to validation set."""

def move_train_to_val(
    raw_dir: Path,
    frac: float = typer.Option(0.1, help="Fraction of training data to move to validation"),
):
    # Hard-coded seed for reproducibility
    seed = 42
    random.seed(seed)

    if not (0.0 < frac < 1.0):
        raise ValueError("frac must be between 0 and 1")

    train_dir = raw_dir / "train"
    val_dir   = raw_dir / "val"

    # collect class subdirs (e.g., NORMAL, PNEUMONIA)
    class_dirs = [p for p in train_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class subfolders found under {train_dir}")

    moves = []
    for cls in class_dirs:
        files = list(cls.rglob("*.jpeg"))
        k = int(len(files) * frac)
        chosen = random.sample(files, k) if k > 0 else []
        for src in chosen:
            rel = src.relative_to(train_dir)  
            dst = val_dir / rel               
            moves.append((src, dst))

    print(f"Planned moves: {len(moves)} files (frac={frac}, seed={seed})")
    # show a few examples
    for i, (src, dst) in enumerate(moves[:10]):
        print(f"  {src} -> {dst}")

    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        # move file
        shutil.move(str(src), str(dst))

    print("Done moving files.")


if __name__ == "__main__":
    typer.run(move_train_to_val)

