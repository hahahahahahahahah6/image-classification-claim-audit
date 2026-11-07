import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from image_classification_claim_audit.audit import audit, evidence, main, matches, IMAGECLASSIFICATION, NOIMAGECLASSIFICATION, GENERIC_NAMES, TORCHVISION_NAMES, KERAS_NAMES, TRANSFORMERS_NAMES

class AuditTests(unittest.TestCase):
    def test_imageclassification_oracle(self):
        self.assertEqual(list(IMAGECLASSIFICATION), ['image classification', 'image-classification', 'imageclassification', 'classify images', 'classifies images', 'classifying images', 'classified images', 'classify image', 'classifies image', 'classifying image', 'image classifier', 'image classifiers', 'image classification support', 'supports image classification', 'image classification capability', 'image classification feature', 'image classification pipeline', 'image classification system', 'image classification tool', 'image classification framework', 'image classification library', 'image classification api', 'image classification model', 'image classification models', 'image classification engine', 'image classification module', 'image classification function', 'image classification functionality', 'image classification workflow', 'image classification method', 'image classification approach', 'image classification service', 'image classification results', 'image classification performance', 'image classification task', 'image classification tasks', 'image classification algorithm', 'image classification algorithms', 'image classification configuration', 'image classification script', 'image classification mode', 'image classification dataset', 'visual classification', 'visual-classification', 'classify pictures', 'classifies pictures', 'classify photos', 'classifies photos', 'photo classification', 'picture classification'])

    def test_noimageclassification_oracle(self):
        self.assertEqual(list(NOIMAGECLASSIFICATION), ['does not classify images', 'do not classify images', "doesn't classify images", "don't classify images", 'no image classification', 'without image classification', 'never classifies images', 'does not support image classification', 'image classification not supported', 'image classification unsupported', 'no image classification support', 'lacks image classification', 'missing image classification', 'image classification disabled', 'no image classification model', 'without image classification support', 'does not provide image classification', 'no automatic image classification', 'image classification not available', 'no image classification capability', 'does not perform image classification', 'never performs image classification', 'does not classify pictures', 'without an image classification model'])

    def test_generic_names_oracle(self):
        self.assertEqual(list(GENERIC_NAMES), ['classify_image', 'image_classify', 'classify_images', 'image_classifier', 'predict_image_class', 'image_class_prediction', 'classify_picture', 'picture_classify', 'label_image', 'image_label', 'predict_image_label', 'get_image_class', 'image_classification_predict', 'classify_photo', 'photo_classify', 'recognize_image_class', 'image_class_recognizer', 'predict_class', 'classify_visual', 'visual_classify', 'image_categorize', 'categorize_image', 'image_tag_predict', 'predict_image_category'])

    def test_torchvision_names_oracle(self):
        self.assertEqual(list(TORCHVISION_NAMES), ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14', 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'googlenet', 'inception_v3', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'squeezenet1_0', 'squeezenet1_1', 'regnet_y_400mf', 'regnet_y_800mf', 'swin_t', 'swin_s', 'swin_b', 'mnasnet0_5', 'mnasnet1_0'])

    def test_keras_names_oracle(self):
        self.assertEqual(list(KERAS_NAMES), ['ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'MobileNet', 'MobileNetV2', 'MobileNetV3Large', 'MobileNetV3Small', 'VGG16', 'VGG19', 'InceptionV3', 'InceptionResNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'Xception', 'NASNetMobile', 'NASNetLarge', 'ConvNeXtTiny', 'ConvNeXtSmall', 'ConvNeXtBase', 'ConvNeXtLarge', 'ConvNeXtXLarge'])

    def test_transformers_names_oracle(self):
        self.assertEqual(list(TRANSFORMERS_NAMES), ['AutoModelForImageClassification', 'ViTForImageClassification', 'ResNetForImageClassification', 'ConvNextForImageClassification', 'EfficientNetForImageClassification', 'BeitForImageClassification', 'DeiTForImageClassification', 'SwinForImageClassification'])

    def fixture(self, readme='', code='', extra=None, flags=()):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'README.md').write_text(readme)
            (root/'app.py').write_text(code)
            for name, text in (extra or {}).items():
                p = root/name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                status = main(['audit', str(root)] + list(flags))
            return status, out.getvalue()

    def test_inventory_matching(self):
        for phrase in IMAGECLASSIFICATION:
            with self.subTest(phrase=phrase):
                self.assertEqual(matches(phrase, IMAGECLASSIFICATION)[0], [phrase])
        for phrase in NOIMAGECLASSIFICATION:
            with self.subTest(phrase=phrase):
                denials, rest = matches(phrase, NOIMAGECLASSIFICATION)
                self.assertEqual(denials, [phrase])
                self.assertEqual(matches(rest, IMAGECLASSIFICATION)[0], [])
        self.assertEqual(matches('image classification model', IMAGECLASSIFICATION)[0], ['image classification model'])

    def test_generics(self):
        for name in GENERIC_NAMES:
            for code in ('x.'+name+'()', 'from unknown import '+name+' as run; run()'):
                with self.subTest(code=code): self.assertTrue(evidence(code))

    def test_split_modules(self):
        self.assertEqual(self.fixture('image classification', 'from torchvision.models import resnet50; m=resnet50()', {'other.py': 'm.forward(x)'})[0], 2)

    def test_matching(self):
        self.assertEqual(matches('image classification', IMAGECLASSIFICATION)[0], ['image classification'])
        self.assertEqual(self.fixture('does not classify images')[0], 0)
        self.assertEqual(self.fixture('IMAGE CLASSIFICATION')[0], 2)
        self.assertEqual(self.fixture('```text\nimage classification\n```')[0], 0)
        self.assertEqual(self.fixture('`image classification`')[0], 2)
        self.assertEqual(matches('we classify images by hand in the photo library', IMAGECLASSIFICATION)[0], ['classify images'])

    def test_cli_good(self):
        self.assertEqual(self.fixture('image classification', 'x.classify_image()'), (0, 'PASS\n'))

    def test_cli_bad(self):
        status, out = self.fixture('image classification')
        self.assertEqual(status, 2)
        self.assertIn('HIGH documented-image-classification-claim-false', out)

    def test_cli_test_only(self):
        extra = {'tests/sample.md': 'image classification'}
        status, out = self.fixture(extra=extra)
        self.assertEqual(status, 0)
        self.assertIn('MEDIUM', out)
        self.assertEqual(self.fixture(extra=extra, flags=['--fail-on', 'medium'])[0], 2)

    def test_cli_denial(self):
        status, out = self.fixture('no image classification', 'x.classify_image()')
        self.assertEqual(status, 2)
        self.assertIn('HIGH documented-no-image-classification-claim-false', out)

    def test_cli_no_claim(self):
        self.assertEqual(self.fixture('A useful tool.'), (0, 'PASS\n'))

    def test_cli_json(self):
        status, out = self.fixture('image classification', flags=['--json'])
        f = json.loads(out)[0]
        self.assertEqual(status, 2)
        self.assertEqual(f['verdict'], 'FAIL')
        self.assertTrue(f['message'])

    def test_cli_ignore(self):
        self.assertEqual(self.fixture(extra={'docs/claim.md': 'image classification'}, flags=['--ignore', 'docs/'])[0], 0)

    def test_cli_version(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as cm:
            main(['--version'])
        self.assertEqual(cm.exception.code, 0)
        self.assertEqual(out.getvalue().strip(), '0.1.0')

    def test_readme(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(self.fixture((root/'README.md').read_text()), (0, 'PASS\n'))

    def test_io_and_usage(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['audit', '/nonexistent-r142-directory']), 1)
            with self.assertRaises(SystemExit) as cm: main(['audit', '--unknown'])
        self.assertEqual(cm.exception.code, 1)

    def test_virtualenv_excluded(self):
        status, out = self.fixture('image classification', extra={
            'custom-env/pyvenv.cfg': 'home = /usr/bin',
            'custom-env/lib/model.py': 'x.classify_image()',
        })
        self.assertEqual(status, 2)

    def test_verbose(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(self.fixture(flags=['--verbose'])[0], 0)
        self.assertIn('scope:', err.getvalue())

    def test_model_pairs(self):
        groups = [('transformers', TRANSFORMERS_NAMES, ('', '.from_pretrained'), ('(x)', '.forward(x)')),
                  ('torchvision.models', TORCHVISION_NAMES, ('',), ('(x)', '.forward(x)')),
                  ('timm', ('create_model',), ('',), ('(x)', '.forward(x)')),
                  ('keras.applications', KERAS_NAMES, ('',), ('.predict(x)',)),
                  ('tensorflow.keras.applications.resnet', ('ResNet50',), ('',), ('.predict(x)',))]
        for module, names, suffixes, runs in groups:
            for name in names:
                for setup, expression in [
                    (f'from {module} import {name}; ', name),
                    (f'from {module} import {name} as F; ', 'F'),
                    (f'import {module} as lib; ', 'lib.' + name),
                    (f'import {module}; ', module + '.' + name),
                ]:
                    for suffix in suffixes:
                        construct = expression + suffix + '()'
                        for run in runs:
                            with self.subTest(module=module, name=name, setup=setup, suffix=suffix, run=run):
                                self.assertTrue(evidence(setup + 'm=' + construct + '; m' + run))
                                self.assertTrue(evidence(setup + construct + run))
                                self.assertTrue(evidence(setup + 'm=' + construct + '; alias=m; alias' + run))
                        self.assertFalse(evidence(setup + construct))
                        self.assertFalse(evidence(setup + 'm=' + construct + '; other(x)'))
                for definition in ('def '+name+'(): pass', 'class '+name+': pass'):
                    self.assertFalse(evidence(f'from {module} import {name}\n{definition}\n{name}()'+runs[0]))
                    self.assertFalse(evidence(f'import {module} as lib\n{definition}\nlib.{name}()'+runs[0]))

    def test_pipeline(self):
        for setup, factory in [('from transformers import pipeline; ', 'pipeline'), ('from transformers import pipeline as P; ', 'P'), ('import transformers as t; ', 't.pipeline'), ('import transformers; ', 'transformers.pipeline')]:
            self.assertTrue(evidence(setup+factory+'("image-classification")(x)'))
            self.assertTrue(evidence(setup+'m='+factory+'("image-classification"); m(x)'))
            self.assertFalse(evidence(setup+factory+'("image-classification")'))
            self.assertFalse(evidence(setup+'m='+factory+'("image-classification"); m.forward(x)'))
            self.assertFalse(evidence(setup+'m='+factory+'("image-classification"); m.direct(x)'))
            self.assertFalse(evidence(setup+factory+'(task="image-classification")(x)'))
            for task in ('video-classification', 'object-detection', 'text-classification', 'image-to-text', 'IMAGE-CLASSIFICATION'):
                self.assertFalse(evidence(setup+factory+'("'+task+'")(x)'))
        self.assertFalse(evidence('from transformers import pipeline\ndef pipeline(x): pass\npipeline("image-classification")(x)'))

    def test_detection_constructors_rejected(self):
        names = ['fasterrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn', 'retinanet_resnet50_fpn', 'ssd300_vgg16', 'deeplabv3_resnet50', 'fcn_resnet50', 'lraspp_mobilenet_v3_large']
        for name in names:
            self.assertNotIn(name, TORCHVISION_NAMES)
            for module in ('torchvision.models', 'torchvision.models.detection', 'torchvision.models.segmentation'):
                self.assertFalse(evidence(f'from {module} import {name}; m={name}(); m(x); m.forward(x)'))

    def test_rejections(self):
        cases = [
            'import torchvision, timm, keras, tensorflow, transformers',
            'from ultralytics import YOLO; YOLO().predict(x)',
            'from detectron2.engine import DefaultPredictor; DefaultPredictor()(x)',
            'import segmentation_models_pytorch as smp; smp.Unet()(x)',
            'import face_recognition as f; f.face_locations(x); f.face_encodings(x)',
            'import cv2; cv2.CascadeClassifier().detectMultiScale(x)',
            'import pytesseract; pytesseract.image_to_string(x)',
            'from easyocr import Reader; Reader().readtext(x)',
            'from transformers import pipeline; pipeline("image-to-text")(x)',
            'from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained(x)(x)',
            'from realesrgan import RealESRGANer; RealESRGANer().enhance(x)',
            'from sklearn.ensemble import RandomForestClassifier; m=RandomForestClassifier(); m.fit(X,y); m.predict(X)',
            'from sklearn.linear_model import LogisticRegression; LogisticRegression().fit(X,y).predict(X)',
            'from sklearn.svm import SVC; SVC().fit(X,y).predict(X)',
            'from transformers import pipeline; pipeline("text-classification")(x)',
            'm(x); m.forward(x); m.predict(x)',
            'from keras.applications import ResNet50; ResNet50()(x)',
            'from keras.applications import ResNet50; ResNet50().forward(x)',
            '"image classification classify images" # image classification',
            '"""classify images"""',
            'from unknown import classify_image; classify_image',
            'classify_image()',
            'import subprocess; subprocess.run(["classification-cli", "image"])',
            'this is invalid python!',
            'from unknown import classify_image\ndef classify_image(): pass\nclassify_image()',
            'from unknown import classify_image\nclass classify_image: pass\nclassify_image()',
            'def classify_image(): pass\nx.classify_image()',
            'from torchvision.models import resnet50\ndef resnet50(): pass\nresnet50()(x)',
            'from torchvision.models import resnet50 as F\ndef F(): pass\nF()(x)',
            'import timm\ndef timm(): pass\ntimm.create_model()(x)',
            'from transformers import AutoModelForImageClassification as M\nclass M: pass\nM.from_pretrained(x)(x)',
        ]
        for code in cases:
            with self.subTest(code=code): self.assertFalse(evidence(code))

    def test_split_all_pairs(self):
        for construction, run in [
            ('from transformers import pipeline; m=pipeline("image-classification")', 'm(x)'),
            ('from transformers import ViTForImageClassification; m=ViTForImageClassification()', 'm.forward(x)'),
            ('from torchvision.models import resnet50; m=resnet50()', 'm(x)'),
            ('import timm; m=timm.create_model()', 'm(x)'),
            ('from keras.applications import ResNet50; m=ResNet50()', 'm.predict(x)'),
        ]:
            self.assertEqual(self.fixture('image classification', construction, {'other.py': run})[0], 2)

    def test_docs_scope_and_skips(self):
        self.assertEqual(self.fixture(extra={'docs/deep/claim.md': 'image classification'})[0], 2)
        self.assertEqual(self.fixture(extra={'other.md': 'image classification'})[0], 0)
        for folder in ('.git', '__pycache__', '.venv', 'venv', 'env'):
            self.assertEqual(self.fixture('image classification', extra={folder+'/app.py': 'x.classify_image()'})[0], 2)

if __name__ == '__main__':
    unittest.main()
