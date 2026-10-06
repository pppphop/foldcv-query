# Third-party sources and data

## Methods and source conventions

JAWS-X: Drew Prinster, Suchi Saria, Anqi Liu and contributors, ICML 2023.
https://proceedings.mlr.press/v202/prinster23a.html
https://github.com/drewprinster/jaws-x/tree/a5c347be096af38832c427d0bf4b702429dc92ba
Its GPLv2 source defines the statistical construction and discrete-tail
conventions reproduced by this software. FoldCV-Query does not claim these
constructions as new. JAWS-X itself is not bundled in this release; its source
is available at the pinned URL. GPLv2 is also the selected license for our code.

## Bundled comparison implementations

FavMac / QuantileTree: Zhen Lin and contributors, copyright 2023 Zhen Lin,
MIT, revision 0130e63b86ac0092b4359bb9993113b4956c87ea.
https://github.com/zlin7/FavMac
Original source and MIT license are retained under
reproduction/standalone/upstream/FavMac.
Lin, Trivedi, Xiao and Sun, ICML 2023:
https://proceedings.mlr.press/v202/lin23j.html

cfcausal: copyright 2019 Lihua Lei, MIT,
revision e4ca6706599873505e1cce2d8cf0078ceff5d2e6, version 0.2.1.
https://github.com/lihualei71/cfcausal
Original R prediction and helper sources, the original copyright file and
MIT license terms are retained under reproduction/standalone/upstream/cfcausal.

## Data provenance

- SGEMM: Enrique Paredes and Rafael Ballester-Ripoll, UCI, CC BY 4.0,
  https://doi.org/10.24432/C5MK70 . Experiment inputs average the four runtimes
  and transform targets with log1p.
- Bike Sharing: Hadi Fanaee-T, UCI, CC BY 4.0,
  https://doi.org/10.24432/C5W894 . Experiment inputs use weather/calendar
  covariates and log1p targets.
- California Housing: R. Kelley Pace and Ronald Barry (1997),
  https://doi.org/10.1016/S0167-7152(96)00140-X .
  Original data are obtained separately through scikit-learn's public
  fetch_california_housing interface; the raw dataset is not redistributed
  here or relicensed as an original FoldCV-Query dataset. The supplied
  fetch_application_data.py writes the runner's X/y input archive.

Saved numerical states and query outputs are derived experiment artifacts;
the underlying sources above must also be credited when reusing them.
Dependency libraries and compilers are installed separately and are not bundled.
