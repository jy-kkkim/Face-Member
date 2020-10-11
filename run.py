import cv2
from os import listdir
from os.path import isdir, isfile, join

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi  # 검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달


def run():
    cap = cv2.VideoCapture(0)
    print("Runnig!!")

    data_path = "models/"
    models_dirs = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    while True:
        ret, frame = cap.read()
        image, face = face_detector(frame)

        try:
            min_score = 999
            min_score_name = ""

            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            model = cv2.face.LBPHFaceRecognizer_create()

            for model_name in models_dirs:
                #print(data_path+model_name)

                model.read(data_path+model_name)
                result = model.predict(face)

                if min_score > result[1]:
                    min_score = result[1]
                    min_score_name = model_name


            if min_score < 500:
                confidence = int(100*(1-(min_score)/300))
                display_string = str(confidence) + '% Confidence it is ' + min_score_name
            cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)
            # 75 보다 크면 동일 인물로 간주해 UnLocked!
            if confidence > 80:
                cv2.putText(image, "Unlocked : " + min_score_name, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', image)
            else:
                # 75 이하면 타인.. Locked!!!
                cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)
        except:
            # 얼굴 검출 안됨
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            pass
        if cv2.waitKey(1) == 13:
            break
    cap.release()
    cv2.destroyAllWindows()





    # model = cv2.face.LBPHFaceRecognizer_create()
    # model.read(data_path)

    # cap = cv2.VideoCapture(0)
    # print("Runnig!!")

    # while True:
    #     # 카메라로 부터 사진 한장 읽기
    #     ret, frame = cap.read()
    #     # 얼굴 검출 시도
    #     image, face = face_detector(frame)
    #     try:
    #         # 검출된 사진을 흑백으로 변환
    #         face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    #
    #         result = model.predict(face)
    #         if result[1] < 500:
    #             # ????? 어쨋든 0~100표시하려고 한듯
    #             confidence = int(100 * (1 - (result[1]) / 300))
    #             # 유사도 화면에 표시
    #             display_string = str(confidence) + '% ung'
    #         cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)
    #         # 75 보다 크면 동일 인물로 간주해 UnLocked!
    #         if confidence > 80:
    #             cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    #             cv2.imshow('Face Cropper', image)
    #         else:
    #             # 75 이하면 타인.. Locked!!!
    #             cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    #             cv2.imshow('Face Cropper', image)
    #
    #     except:
    #         # 얼굴 검출 안됨
    #         cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    #         cv2.imshow('Face Cropper', image)
    #         pass
    #     if cv2.waitKey(1) == 13:
    #         break
    # cap.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    run()