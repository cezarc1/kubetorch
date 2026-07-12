# Kubetorch Tutorial Metadata Cleanup Design

## Objective

Remove defensive validation and adaptation language from the public tutorial
experience. Tutorials should lead with the material itself, not with caveats
about fork versions, static checks, or unrecorded cluster runs.

The validation catalog and smoke-evidence machinery remain intact as internal
maintenance metadata.

## Public presentation

Each generated tutorial begins with its title followed by one neutral line:

```text
Requires: GPU · View source
```

The line retains the tutorial's existing hardware or infrastructure
requirements and source link. It does not expose validation state, fork
version, execution status, or a colored status badge.

The tutorial landing page no longer explains **Validated**, **Adapted**, or
**Reference**. Repository-facing tutorial READMEs no longer advertise those
states as reader guidance. Ordinary uses of words such as "reference model" or
"adapted dataset" inside technical source remain unchanged.

## Implementation

- Replace the renderer's status-admonition function with a neutral metadata
  line generator.
- Regenerate all 33 checked-in tutorial pages from their source.
- Remove the validation-badge explanation from the tutorial landing page.
- Remove now-unused `.kt-status*` CSS rules.
- Simplify tutorial-maintenance READMEs while retaining upstream attribution,
  pinned-source provenance, catalog location, and generation commands.
- Keep catalog validation fields, schema checks, smoke commands, and evidence
  generation unchanged.

## Verification

- Update renderer tests to assert requirements and source links remain while
  defensive wording is absent.
- Add a repository-level assertion that generated tutorial introductions do not
  contain status admonitions or the removed phrases.
- Run the renderer drift check and complete docs test suite.
- Build Sphinx with warnings treated as errors.
- Refresh the local preview and inspect the tutorial landing page, MNIST, basic
  GRPO, VeRL, and one external-infrastructure tutorial.

No branch push, pull request, or Pages deployment occurs until the user approves
the local preview.
