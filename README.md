# rtc-tools-diagnostics

This is rtc-tools-diagnostics, a toolbox to analyse results from [rtc-tools](https://gitlab.com/deltares/rtc-tools).

## Install

```bash
pip install rtc-tools-diagnostics
```
## Features
### Export results after each priority
The `ExportResultsEachPriorityMixin` enables RTC-Tools to save the timeseries export after solving each priority. In addition to the usual final timeseries_export file (.csv or .xml) found in the output folder, a separate folder
will be created for each priority. Each priority folder will contain the results for the problem considering all goals up to and including that priority. To use this functionality, import the mixin with the following code:
```python
from rtctools_diagnostics.export_results import ExportResultsEachPriorityMixin
```
and add the `ExportResultsEachPriorityMixin` to you optimization problem class, before the other rtc-tools classes and mixins included in optimization problem class.

You can disable this functionality by setting the class variable `export_results_each_priority` to `False` from your optimization problem class.

### Get optimization problem formulation and active constraints
By using the `GetLinearProblemMixin`, you can generate a file that indicates the active constraints and bounds for each priority. To utilize this functionality, import the mixin as follows:
```python
from rtctools_diagnostics.get_linear_problem import GetLinearProblemMixin
```
Then, add the `GetLinearProblemMixin` to your optimization problem class, before the other rtc-tools classes and mixins included in optimization problem class. After running the model, a file named `active_constraints.md` will be available in your output folder.

### Save LP file of the optimization model

If an RTC-Tools model inherits from `ExtractLPMixin`, an LP file is generated and stored in the output folder, representing the optimization model that is passed to the solver. For each priority, a separate LP file is created.

To use this functionality, first import it as follows:

```python
from rtctools_diagnostics.get_linear_problem import ExtractLPMixin
```
Then, inherit from `ExtractLPMixin`. Note that this class should come before any other mixin class in your inheritance list.


### Notes
- MIP and non-linear problems are not supported (yet) by `GetLinearProblemMixin` and `ExtractLPMixin`.