# Each kind is a module with main(run_dir): reads run_dir/params.json, writes run_dir/result.json.

KINDS = {"noop", "eval", "prep", "train", "export", "synth", "fetch"}
