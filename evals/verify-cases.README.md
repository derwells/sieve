# Citation verification cases

`verify-cases.json` contains 40 hand-written claim/evidence records for `jev_verify`. `v01`–`v20` are the fit set; `v21`–`v40` are the held-out test set. Each half has 10 faithful claims and 10 altered claims, with two each of `number`, `negation`, `scope`, `date`, and `wrong_citation`. Four scope cases combine a supported clause with an unsupported one and expect `partially_supports`.

Every citation has one verbatim 8–40-word quote from its locator. TypeSafe documentation locators use the public `.md` endpoint; `README.md` refers to the repository root. The expected verdict judges the cited passage against the claim. A `wrong_citation` claim may be true elsewhere, but its cited passage does not address it. `claim_context` names the subject of the claim; `notes` explains the edit and expected judgment.

The quotes were checked as exact substrings of the fetched markdown or local README before writing the dataset. These are hand-labeled expectations, not observed checker outputs. Public docs may change, so recheck quote matches after updating them.
