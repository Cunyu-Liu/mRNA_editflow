# Autoresearch classic loop: mRNA-EditFlow

Primary SetFlow acceptance signals:

- common Validation NLL <= 2.06809;
- source-macro recovery >= 0.35;
- source-macro top-k recovery >= 0.20;
- source-macro unique candidate rate >= 0.90;
- legality 1.0 and all replay, budget, and numerical failure counts equal 0;
- full model recovery margin over single-mode >= 0.03 and unique margin >= 0.05.

Verification is the terminal recovery `screen_gate.json` plus all eight
`validation_summary.json` artifacts. Development TEST and new final Evaluation
outcomes remain unread during this loop.
