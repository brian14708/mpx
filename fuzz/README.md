# Fuzzing

Requires the toolchain in `rust-toolchain.toml` and cargo-fuzz (`nix develop`
provides both).

From the repository root:

```sh
python3 scripts/fuzz.py --jobs 4                         # seed, build, replay all corpora
python3 scripts/fuzz.py --seconds 60 --jobs 4            # fuzz each target for 60 seconds
python3 scripts/fuzz.py protocol_stream --seconds 60     # fuzz one target
cargo fuzz run -s none TARGET fuzz/artifacts/TARGET/crash-HASH  # reproduce a crash
```

`cargo fuzz list` shows the targets. Seeds come from `src/seeds.rs`, and input
layouts are documented in `src/protocol_input.rs` and the target sources.
