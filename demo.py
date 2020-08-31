import argparse
import better_exceptions
from pathlib import Path
from contextlib import contextmanager
import urllib.request
import facenet_pytorch
import numpy as np
import cv2
# import dlib
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from model import get_model
from defaults import _C as cfg
from time import perf_counter
from facenet_pytorch import MTCNN
from PIL import Image
from dataset import expand_bbox
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_args():
    parser = argparse.ArgumentParser(description="Age estimation demo",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resume", type=str, required = True,
                        help="Model weight to be tested")
    parser.add_argument("--img_dir", type=str, default=None,
                        help="Target image directory; if set, images in image_dir are used instead of webcam")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory to which resulting images will be stored if set")
    parser.add_argument('--ldl', action="store_true",
                        help="Use KLDivLoss + L1 Loss")
    parser.add_argument('--expand', type=float, default=0, help="expand the crop area by a factor, typically between 0 and 1")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img, None


def yield_images_from_dir(img_dir):
    img_dir = Path(img_dir)

    for img_path in img_dir.glob("*.*"):
        img = cv2.imread(str(img_path), 1)

        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r))), img_path.name


def angel(t1, t2):
    x = t2[0]-t1[0]
    y = (t2[1]-t1[1])
    return math.degrees(math.atan2(y, x))


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()

    if args.output_dir is not None:
        if args.img_dir is None:
            raise ValueError("=> --img_dir argument is required if --output_dir is used")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = get_model(model_name=cfg.MODEL.ARCH, pretrained=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # load checkpoint
    resume_path = args.resume

    if resume_path is None:
        resume_path = Path(__file__).resolve().parent.joinpath("misc", "epoch044_0.02343_3.9984.pth")

        if not resume_path.is_file():
            print(f"=> model path is not set; start downloading trained model to {resume_path}")
            url = "https://github.com/yu4u/age-estimation-pytorch/releases/download/v1.0/epoch044_0.02343_3.9984.pth"
            urllib.request.urlretrieve(url, str(resume_path))
            print("=> download finished")

    if Path(resume_path).is_file():
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(resume_path))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))

    if device == "cuda":
        cudnn.benchmark = True

    model.eval()
    img_dir = args.img_dir
    # detector = dlib.get_frontal_face_detector()
    mtcnn = MTCNN(device=device, post_process=False, keep_all=False)
    img_size = cfg.MODEL.IMG_SIZE
    image_generator = yield_images_from_dir(img_dir) if img_dir else yield_images()
    rank = torch.Tensor([i for i in range(101)]).to(device)

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()
    ])

    with torch.no_grad():
        for img, name in image_generator:
            start = perf_counter()
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(input_img)
            detected, probs = mtcnn.detect(image)
            img_h, img_w = image.size

            # # detect faces using dlib detector
            # detected = detector(input_img, 1)
            # print(detected)

            if detected is not None and len(detected) > 0:
                detected = detected.astype(int)
                if args.expand > 0:
                    box = expand_bbox(image.size, detected[0], ratio= args.expand)
                else:
                    box = detected[0]
                image = image.crop(box)
                cv2.rectangle(img, (detected[0][0], detected[0][1]), (detected[0][2], detected[0][3]), (255, 255, 255), 2)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

                image_np = np.array(image)
                augmented = transform(image = image_np)
                image = augmented["image"]
                image = image.unsqueeze(0).to(device)
                # print(image.shape)

            #     # predict ages
                outputs = model(image)
                outputs = F.softmax(outputs, dim=1)
                if args.ldl:
                    predicted_ages = torch.sum(outputs * rank, dim = 1)
                else:
                    _, predicted_ages = outputs.max(1)

                # draw results
                # for i, d in enumerate(detected):
                label = "{}".format(int(predicted_ages[0]))
                draw_label(img, (detected[0][0], detected[0][1]), label)

                # faces = np.array(faces.permute(1, 2, 0)).astype(np.uint8)
                # faces = cv2.cvtColor(faces, cv2.COLOR_RGB2BGR)

            if args.output_dir is not None:
                output_path = output_dir.joinpath(name)
                cv2.imwrite(str(output_path), img)
            else:
                elapsed = perf_counter() - start
                cv2.putText(img, "FPS: "+ "{:.1f} Press ESC to exit".format(60/elapsed),(10,20),cv2.FONT_HERSHEY_DUPLEX,0.5,(255, 255,255),1)
                cv2.imshow("result", img)
                key = cv2.waitKey(-1) if img_dir else cv2.waitKey(30)

                if key == 27:  # ESC
                    break


if __name__ == '__main__':
    main()
