"""Build the gap-check puzzle set: 20 Reasoning Gym tasks x N items, default difficulty, seed 20261108."""
import json, sys, reasoning_gym as rg
TASKS = ["knights_knaves","zebra_puzzles","futoshiki","mini_sudoku","countdown","puzzle24","cryptarithm",
         "graph_color","shortest_path","maze","course_schedule","propositional_logic","self_reference",
         "circuit_logic","jugs","kakurasu","survo","arc_1d","number_sequence","binary_matrix"]
N, SEED = int(sys.argv[1]), 20261108
out = open(sys.argv[2], "w")
for i, t in enumerate(TASKS):
    ds = rg.create_dataset(t, size=N, seed=SEED + i)
    ok = 0
    for j, e in enumerate(ds):
        s = ds.score_answer(e["answer"], e)
        ok += s >= 0.999
        out.write(json.dumps({"id": f"{t}-{j}", "task": t, "seed": SEED + i, "idx": j,
                              "question": e["question"], "answer": e["answer"]}) + "\n")
    print(f"{t:22s} gold self-score {ok}/{N}")
