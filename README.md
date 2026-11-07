# Documentation claim auditor

Offline, deterministic, standard-library-only inspection of documentation about
whole-image labelling. Requires Python 3.9 or later. No network, telemetry, model
execution, or file writes are performed.

## Usage

```sh
python3 -m pip install .
image-classification-claim-audit audit .
python3 -m image_classification_claim_audit audit . --json
image-classification-claim-audit audit . --verbose --ignore vendor/ --ignore examples/
image-classification-claim-audit audit . --fail-on medium
image-classification-claim-audit --version
```

The repository defaults to the current directory. Exit status is 0 when no
finding reaches the threshold, 2 when one does, and 1 for usage or IO errors.
The default threshold is high. JSON emits a list of findings with id, severity,
path, verdict, message, and phrases. Verbose scope and per-file decisions go to
standard error. Repeatable ignore prefixes match repository-relative paths.

## Scope and rules

Scans the root README and Markdown below docs. Claims in files under a tests
directory receive medium severity. Python sources are inspected throughout the
repository, excluding version control, bytecode caches, virtual environments,
and build output. Each evidence route must be complete in one module. Imported
aliases, simple assigned aliases, and chained calls are supported. Local function
and class definitions shadow imported names. Unparseable modules supply no evidence.

Fenced code is excluded; inline code counts as prose. Matching uses exact,
case-insensitive substrings without stemming or hyphen normalization. Denial
spans are removed first. Longest-first matching blanks spans so each occurrence
matches once under its longest phrase. A claim without evidence is high severity;
a denial contradicted by evidence is high severity. A denial without evidence
passes. Findings use these identifiers:

```text
documented-image-classification-claim-false
documented-no-image-classification-claim-false
```

## Differentiation

The following neighbor checks generic label prediction. This auditor checks
whole-image labelling; a generic classifier fitted on tabular features is
deliberately not counted because it classifies something else, not images.

```text
classification-claim-audit
```

The next neighbor checks localising and labelling regions, whereas the checked
feature assigns one label to the whole image.

```text
object-detection-claim-audit
```

The next neighbor checks finding faces, whereas this auditor checks labelling
the whole image.

```text
face-claim-audit
```

The next neighbor checks describing an image in words, whereas this auditor
checks assigning it a label.

```text
image-captioning-claim-audit
```

## Evidence and limitations

Static pattern matching is not semantic judgement. Exact phrases miss restatement.
Control flow and runtime types are not evaluated. No model quality or prediction
correctness is verified. The following pipeline construction requires a call on
its returned instance in the same module. The pair counts as evidence although
it alone does not prove the produced label is correct or useful.

```python
from transformers import pipeline
p = pipeline("image-classification")
p(image)
```

The following model family requires construction, directly or through its
pretrained loader, plus a direct or forward call on the returned instance.

```text
transformers: AutoModelForImageClassification, ViTForImageClassification,
ResNetForImageClassification, ConvNextForImageClassification,
EfficientNetForImageClassification, BeitForImageClassification,
DeiTForImageClassification, SwinForImageClassification
```

A construction in the next family paired with a forward or direct call shows
the feature is used, not that the produced label is correct. Only the explicit
constructor inventory is accepted; detection and segmentation models do not count.

```python
from torchvision.models import resnet50
m = resnet50()
m.forward(image)
```

The following factory also requires a direct or forward call on its returned
instance in the same module.

```python
import timm
m = timm.create_model("resnet50")
m(image)
```

An applications construction in the following family paired with a predict call
counts as evidence although the call alone does not prove the produced label is
correct. Assigned and chained forms are accepted.

```python
from keras.applications import ResNet50
ResNet50().predict(image)
```

A generic call on an unrelated object counts via the generic route because the
method name is shared. Bare generic calls require a matching from-import.

```python
unrelated.classify_image(image)
```

The first three phrases below can also describe a person sorting photos by hand.
The next two can describe a person organising a photo album, and the final two
can describe manual sorting. Exact substring matching cannot distinguish those
meanings; longest-first matching still counts each occurrence only once.

```text
classify images
classifies images
classifying images
classify pictures
classify photos
photo classification
picture classification
```

Bare imports, strings, comments, docstrings, subprocess commands, construction
alone, unrelated forward or predict calls, and evidence split across modules do
not count. Region finding, pixel labelling, face processing, text reading,
captioning, generation, and upscaling alone do not establish the checked capability.

## Development

```sh
python3 -m unittest discover
python3 -m build
```

Distributed under the MIT license.
