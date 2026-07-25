Read the word in `/ale/input/word.txt` and write a greeting to
`/ale/output/result.txt`.

The greeting must be exactly `${greeting}`, then a space, then the word from the input
file, with no trailing newline.

Paths are written literally on purpose: every task sees the same workspace
(`/ale/input`, `/ale/output`, `/ale/work`), so nothing here depends on this task's name
or on where its folder sits.
