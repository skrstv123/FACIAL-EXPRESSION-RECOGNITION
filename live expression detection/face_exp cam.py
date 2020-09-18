import cv2
import tensorflow as tf 
import numpy as np  
facc=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyec = cv2.CascadeClassifier('haarcascade_eye.xml')
model = tf.keras.models.load_model("expressionmodel.h5")
classes = {'angry': 0,
		 'disgust': 1,
		 'fear': 2,
		 'happy': 3,
		 'neutral': 4,
		 'sad': 5,
		 'surprise': 6
}
cl = {i:j for j,i in classes.items()}
fl = 1 
def draw_rect(bw,col):

	face_rects = facc.detectMultiScale(bw, 1.3, 5)
	for x,y,w,h in face_rects:
		cv2.rectangle(col, (x,y),(x+w,y+h),(0,0,255),4)
		bw_face = bw[y:y+h,x:x+h]
		try:
			org = (y,x) 
			exp = model.predict(np.expand_dims(np.expand_dims(cv2.resize(bw_face, (48,48)), axis=-1), axis=0))
			prd = cl[np.argmax(exp[0])]
			cv2.putText(col, prd, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 50, 0) if prd!='neutral' else (36,255,12) , 2)
			# cv2.putText(col, cl[np.argmax(exp[0])] , org, font,fontScale, color, thickness, cv2.LINE_AA) 
		except : pass

		col_face = col[y:y+h,x:x+h]
		eye_rects = eyec.detectMultiScale(bw_face,1.1,3)
		for a,b,br,le in eye_rects:
			cv2.rectangle(col_face,(a,b),(a+br,b+le),(0,255,0),2)
	return col 
print("press q to exit")
internal_web_cam = cv2.VideoCapture(0)
while 1:
    _unused,clframe = internal_web_cam.read()
    xx= type(clframe)
    bw = cv2.cvtColor(clframe,cv2.COLOR_BGR2GRAY)
    with_rect = draw_rect(bw,clframe)
    cv2.imshow('Video',with_rect)
    if cv2.waitKey(1)& 0xFF == ord('q'):
        # print(xx)
        break
internal_web_cam.release()
cv2.destroyAllWindows()