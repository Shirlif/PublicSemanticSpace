# Claim-gap E5: contract-free baseline vs PSS L0

> seeds: 3, dim: 128

| metric | PSS L0 | flat baseline | PSS > flat |
|---|---:|---:|---|
| SSR | 1.000 | 1.000 | False |
| SRR | 1.000 | 0.000 | True |
| INT | 0.000 | 0.667 | True |
| RBE | 1.000 | 0.000 | True |
| RCR | 1.000 | 0.000 | True |
| USR | 1.000 | 1.000 | True |
| LOC | 0.000 | 0.000 | True |
| PCR | 1.000 | 0.000 | True |
| AUD | 1.000 | 0.000 | True |
| RBE-KB | 1.000 | 0.000 | True |
| BAR | 1.000 | 0.000 | True |
| MON | 1.000 | 0.000 | True |

## How to read this

The contract-free baseline accepts invalid modules, has no rollback or
versioned routing, shares one codebook across modalities, has no audit or
justification propagation, and has no budget controller. The property suite
must mark those as failures while PSS passes them. This is what makes the
suite a measurement instrument rather than a description of PSS.

It is a negative architecture, not a competitive model: it is not a
capability baseline and should not be read as one.
