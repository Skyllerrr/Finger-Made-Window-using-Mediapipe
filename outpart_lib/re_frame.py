import cv2

def go_first_frame (mv):
  TF = mv.get(cv2.CAP_PROP_FRAME_COUNT)
  CF = mv.get(cv2.CAP_PROP_POS_FRAMES)
  if TF == CF:
    mv.set(cv2.CAP_PROP_POS_FRAMES, 0)