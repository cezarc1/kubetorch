# Video Context Recovery and Footer Removal Design

## Goal

Remove the Runhouse author/copyright footer and all visitor-facing YouTube
embeds and links. Before removing the video references, recover any useful
technical explanation that is missing from the maintained documentation.

The final pages will stand on their own as written documentation. Raw
transcripts will be temporary research inputs and will not be committed.

## Source inventory

Nine videos are currently rendered:

| Video ID | Documentation target |
| --- | --- |
| `CH0mMcR5hZ8` | Automatic Batch Size Recovery |
| `k1olO4P_1WY` | Dynamic World Size |
| `5nRRaxZJnUg` | PyTorch Multi-Node Distributed Training |
| `sqNYnowFufY` | RAG Composite Inference |
| `yB29sojkAiU` | TRL with a Code Sandbox |
| `-oz49qt_uSM` | VeRL GRPO Training |
| `VCWpCM2m1Hw` | OpenAI GPT-OSS 120B Inference |
| `8slAR7459X4` | vLLM Inference |
| `9vQww8bhCzY` | Distributed PDB Debugging |

The imported DeepSeek OCR supporting source contains one additional marker for
`yJ3b6Gps9qI`, but it is not rendered as a tutorial page.

## Transcript recovery workflow

Use the English automatic-caption track for each video as a temporary research
source. For every transcript:

1. Compare the spoken explanation with the current tutorial source and
   generated page.
2. Ignore introductions, repetition, marketing claims, obsolete Runhouse
   hosting instructions, and details already explained by the page or code.
3. Verify API names and operational claims against the current fork before
   adding them.
4. Add concise prose only when it clarifies architecture, execution order,
   failure behavior, scaling behavior, or a non-obvious code choice.
5. Do not quote or commit the transcript. Record only the resulting maintained
   explanation.

For catalogued tutorials, durable prose belongs in the literate comments of
the imported example source, followed by regeneration. The hand-authored
debugging guide is edited directly. Relevant DeepSeek OCR details may be added
to its source comments, but this work will not create a new tutorial route.

## Removing video presentation

After the transcript comparison:

- remove `video_id` values from the tutorial catalog;
- remove `::youtube` markers from imported example sources;
- remove the hand-authored YouTube directive from the debugging guide;
- stop the tutorial renderer from emitting YouTube directives;
- remove the unused Sphinx YouTube extension and its tests;
- ensure generated documentation contains no YouTube embeds or fallback links.

An internal recovery record may retain video IDs, page mappings, titles, and
the transcript-review disposition. It will not be included in the Sphinx
navigation or rendered HTML.

## Footer removal

Suppress the Sphinx Book Theme content-footer items so the rendered pages no
longer show either:

- `By the Runhouse team and Kubetorch fork maintainers`
- `© Copyright Runhouse Inc.`

Runhouse project attribution remains in the homepage, repository README, and
maintainer records. The change removes redundant page furniture rather than
erasing project provenance.

## Validation

Tests will verify that:

- all tutorial catalog video fields are empty;
- literate rendering discards legacy video markers without producing an embed;
- no maintained documentation or imported example source contains YouTube
  URLs or directives;
- the theme footer item list is empty;
- regenerated tutorial pages match their sources;
- the Sphinx build succeeds with warnings treated as errors;
- built HTML contains neither YouTube embeds/links nor the removed footer text.

The final local-browser check will sample the footer, debugging guide, and at
least one generated tutorial to confirm the written context renders correctly
and no empty video or footer containers disrupt the layout.
