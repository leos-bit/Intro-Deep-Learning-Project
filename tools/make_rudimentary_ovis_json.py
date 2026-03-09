import argparse
import json
import random
from pathlib import Path


def generate_trajectory(length, x0, y0, w0, h0, vx0, vy0, occlusion_prob):
    x, y, w, h = x0, y0, w0, h0
    vx, vy = vx0, vy0
    bboxes = []

    for _ in range(length):
        if random.random() < occlusion_prob:
            bboxes.append(None)
        else:
            bboxes.append([round(x, 2), round(y, 2), round(max(8.0, w), 2), round(max(8.0, h), 2)])

        ax = random.uniform(-0.6, 0.6)
        ay = random.uniform(-0.6, 0.6)
        vx += ax
        vy += ay
        x += vx
        y += vy
        w += random.uniform(-1.0, 1.0)
        h += random.uniform(-1.0, 1.0)

    return bboxes


def build_rudimentary_ovis(num_videos, frames_per_video, tracks_per_video, occlusion_prob, seed):
    random.seed(seed)

    data = {
        "videos": [],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "car"},
            {"id": 3, "name": "bicycle"},
        ],
        "annotations": [],
    }

    ann_id = 1
    for vid_id in range(1, num_videos + 1):
        data["videos"].append(
            {
                "id": vid_id,
                "name": f"rudimentary_video_{vid_id:03d}",
                "width": 1280,
                "height": 720,
                "length": frames_per_video,
                "file_names": [f"video_{vid_id:03d}/frame_{f:04d}.jpg" for f in range(1, frames_per_video + 1)],
            }
        )

        for _ in range(tracks_per_video):
            category_id = random.choice([1, 2, 3])
            x0 = random.uniform(40, 1000)
            y0 = random.uniform(40, 580)
            w0 = random.uniform(30, 180)
            h0 = random.uniform(30, 180)
            vx0 = random.uniform(-6, 6)
            vy0 = random.uniform(-5, 5)
            bboxes = generate_trajectory(frames_per_video, x0, y0, w0, h0, vx0, vy0, occlusion_prob)

            data["annotations"].append(
                {
                    "id": ann_id,
                    "video_id": vid_id,
                    "category_id": category_id,
                    "iscrowd": 0,
                    "bboxes": bboxes,
                }
            )
            ann_id += 1

    return data


def main():
    parser = argparse.ArgumentParser(description="Create rudimentary OVIS-style JSON annotations")
    parser.add_argument("--out", type=str, default="data/rudimentary_ovis_train.json")
    parser.add_argument("--num-videos", type=int, default=3)
    parser.add_argument("--frames-per-video", type=int, default=20)
    parser.add_argument("--tracks-per-video", type=int, default=4)
    parser.add_argument("--occlusion-prob", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = build_rudimentary_ovis(
        num_videos=args.num_videos,
        frames_per_video=args.frames_per_video,
        tracks_per_video=args.tracks_per_video,
        occlusion_prob=args.occlusion_prob,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote rudimentary OVIS JSON to {out_path.resolve()}")


if __name__ == "__main__":
    main()
