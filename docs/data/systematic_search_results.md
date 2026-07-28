# D0 systematic search results — UTR intervention evidence

Generated: 2026-07-28

Contract: `utr_editflow_goal_v2`

Protocol: `docs/data/systematic_search_protocol.md`

## Source log

| Source | Result |
|---|---|
| PubMed/PMC and publisher primary text | MPRAu design verified; variable-length 5′UTR library designs verified |
| GEO/SRA | Existing accessions rechecked at metadata level; GSE330741 and GSE291719 discovered and retained metadata-only |
| ENCODE/ENA | ENCSR854RUF identity and 62-file raw inventory path retained; active reconstruction not disturbed |
| OpenAlex | Broad title/abstract fallback completed; no additional verified source-paired UTR insertion dataset adopted |
| MaveDB/Zenodo/Figshare/official code | No additional candidate was promoted without primary identity and required source/candidate metadata |

## Qualified findings

### Measured UTR indel

MPRAu is measured 3′UTR evidence and includes non-overlapping 5-bp deletion
tiling across tested 3′UTRs around a subset of 80 tamVars, in addition to
single-nucleotide changes. This qualifies as measured deletion coverage for
that design only. It does not establish insertion coverage or a measured
multi-step trajectory. The processed MPRAu measurement role is kept separate
from the 62 reconstructed raw-read assets: the raw inventory alone is only an
observational/pretraining candidate and cannot be promoted to intervention
evidence.

Primary record:
<https://pmc.ncbi.nlm.nih.gov/articles/PMC8487971/>

### Variable-length libraries

Sample et al. measured random 25-nt and 50-nt 5′UTR libraries and designed
5′UTRs. These data can support an absolute sequence prior or measured
landscape analysis. They do not, without a recoverable explicit source for
each candidate, establish source-paired insertion/deletion transitions.

Primary record:
<https://pubmed.ncbi.nlm.nih.gov/31267113/>

A later 5′UTR study also reports fixed 25-nt and 50-nt random libraries and
designed sequences. It is relevant to absolute design and foundation reuse,
not evidence of observed edit trajectories.

Primary record:
<https://pmc.ncbi.nlm.nih.gov/articles/PMC11189900/>

An earlier zebrafish 3′UTR MPRA measured 90,000 fixed 110-nt sequences
covering annotated UTR fragments. This is a large absolute library and useful
context for sequence priors, but it is neither variable-length nor a
source-paired insertion/deletion trajectory dataset.

Primary record:
<https://pmc.ncbi.nlm.nih.gov/articles/PMC5994907/>

### Dense and combinatorial endpoints

GSE145046 provides a dense designed 5′UTR landscape on a fixed scaffold.
Changed positions may be represented by constructed canonical edit scripts,
but the endpoint library does not observe the order of biological edits.
GSE200304 and the other paired natural-variant sets ground measured
substitutions, not general INS/DEL trajectories.

### New external candidate

GSE330741 became public on 2026-05-17. Its official metadata describes
single-nucleotide MPRA tiling of the Glt1 and Sparc 3′UTRs, with deletion
validation and 163 samples. In V2 only metadata were inspected. The dataset is
therefore a candidate for a future pre-access freeze/exposure audit, not yet an
untouched E5 set; no candidate-level final labels were accessed in D0.

Official record:
<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE330741>

GSE200303/GSE200304 remains an already known measured 3′UTR substitution
resource (6,892 reported mutations), not new indel evidence.

Official record:
<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE200303>

GSE291719 became public on 2025-05-14. Official metadata describes an
approximately 500-member synthetic 3′UTR reporter library across HEK, CD4+
T-cell and CD8+ T-cell contexts, with 27 samples and public raw/normalized
count files. D0 did not access those candidate-level count tables. The
accession is retained as a metadata-only conditional-context candidate pending
source/template reconstruction, license/exposure audit and a label-free freeze;
it is not source-paired INS/DEL evidence and is not yet E5.

Official record:
<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE291719>

## License and reuse audit

- ENCODE states that publicly released data are available for unrestricted
  use; scientific citation remains required practice:
  <https://www.encodeproject.org/about/data-use-policy/>.
- ENA/INSDC states free and unrestricted access and redistribution/use of
  public sequence records, with credit to the original submission:
  <https://www.ebi.ac.uk/ena/browser/about/policies>.
- NCBI places no restriction on GEO data use or distribution, while warning
  that submitters may retain patent, copyright or other rights:
  <https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html>.

The matrix therefore records repository policy and residual submitter-rights
uncertainty rather than assigning an unsupported blanket Creative Commons
license.

## Negative findings retained

- No verified source-paired UTR insertion dataset was found in this D0 search.
- No candidate was verified to provide an experimentally observed multi-step
  UTR edit trajectory.
- Variable-length absolute libraries do not by themselves resolve edit-script
  direction, source identity or transition order.
- The GSE330741 role remains unresolved until a D1 exposure audit and
  label-free freeze; public availability alone does not prove untouched status.
- GSE291719 is a synthetic absolute library pending source/template audit, not
  a verified edit-pair or insertion dataset.
- Source and API coverage is not proof of universal absence. These are bounded
  D0 findings and systematic searching remains an allowed forward path.

## Forward path

Retain measured single-edit grounding; use absolute variable-length data only
for a separately labelled prior; use dense endpoint landscapes only with path
ambiguity recorded; validate unsupported grammar operations synthetically; and
continue metadata-first external discovery. Corresponding biological claims
remain blocked rather than being replaced by predictor-only claims.
