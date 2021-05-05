# elapid

<img src="http://earth-chris.github.io/images/design/amazon.jpg" alt="the amazon"/>

---

**Documentation**: [earth-chris.github.io/elapid](https://earth-chris.github.io/elapid)

**Source code**: [earth-chris/elapid][https://github.com/earth-chris/elapid)

---

## Introduction

`elapid` provides python support for species distribution modeling. This includes a custom [MaxEnt][home-maxent] implementation and general spatial processing tools. It will soon include tools for working with [GBIF][home-gbif]-format datasets.

The name is an homage to *A Biogeographic Analysis of Australian Elapid Snakes* (H.A. Nix, 1986), the paper widely credited with defining the essential bioclimatic variables to use in species distribution modeling. It's also a snake pun (a python wrapper for mapping snake biogeography).

The maxent modeling tools and feature transformations are translations of the R `maxnet` [package][r-maxnet]. It uses the `glmnet` [python bindings][glmnet], and is implemented using `sklearn` conventions.

This package is still in the early stages of development.

## Contact

* Christopher Anderson - [web][home-cba] - [email][email-cba] - [github][github-cba].


[email-cba]: mailto:cbanders@stanford.edu
[github-cba]: https://github.com/earth-chris
[home-cba]: https://earth-chris.github.io
[home-conda]: https://docs.conda.io/
[home-gbif]: https://gbif.org
[home-maxent]: https://biodiversityinformatics.amnh.org/open_source/maxent/
[r-maxnet]: https://github.com/mrmaxent/maxnet
[glmnet]: https://github.com/civisanalytics/python-glmnet/
