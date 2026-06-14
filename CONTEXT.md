# Context glossary

Ubiquitous language for vlm-sink-emergence. Glossary only — no implementation, no specs.

## Terms

- **Arm** — one training run with a fixed lever setting (`train_sinks.py --arm X`). Four
  defined: `baseline` (softmax / random LM / pretrained ViT), `g1gate` (Qiu σ-gate),
  `sigmoid` (unnormalized, no softmax), `textinit` (pretrained SmolLM2 LM). `--vit_init
  random` makes any arm fully-from-scratch.

- **Signature** — a measurable sink fingerprint logged by the probe every run. Three,
  tracked **separately** (their decoupling is the contribution):
  *concentration* (`Sink^ε_1`, mean/max attn→pos0), *value-norm* (‖v‖ pos0 vs rest),
  *massive-activation* (residual norm pos0 vs rest). Not to be confused with arms.

- **RF (random-fresh baseline)** — the `baseline` arm trained on **fresh** FineVision data
  at 1B tokens. Exists to re-test **Gate A** free of the repetition confound. One arm, not
  a sweep. Validity rule: byte-identical recipe + `max_steps` to its **comparator** (the 1B
  repeated Gate-A run), data path the only difference. RF passes Gate A iff Sink^0.3_1 < ~5%
  **and** fresh val_loss healthy (no divergence).

- **Comparator** — the specific ~855M–1B *repeated* run whose confounded Gate-A verdict RF
  replaces. RF clones its `run_config.json`; not the shorter Session-1 174M baseline.

- **Repeated vs Fresh** — *repeated* = the original 4 the_cauldron subsets (~146K images →
  ~74 visual epochs at 1B tok, the overfitting confound). *fresh* = FineVision natural-image
  pool (~4.6M images → ~1–2 epochs at 1B). The fresh-vs-repeated comparison is valid only if
  data is the **sole** difference (identical layout/collator/template/tokenizer).

- **Gate A** — go/no-go: does the concentration sink stay absent (Sink^0.3_1 < ~5%) at the
  token budget from random init? Session-3 verdict CONFIRM but **confounded** by repeated-data
  overfitting; RF is the clean retest.

- **Overfitting confound** — repeated baseline reached ~74 visual epochs, train_loss→0.015,
  val flat/rising. Taints Gate A's "no sink" reading. Does **not** taint the norm/value
  signatures or the 4-way dissociation (survive n=2).
