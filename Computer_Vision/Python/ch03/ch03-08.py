import cv2 as cv

img = cv.imread("/home/kimhoyun/CS_Study/Computer_Vision/Python/img/starship.jpg")
patch = img[250:350, 170:270, :]

img = cv.rectangle(img, (170,250), (270,350), (0,0,255),2)
patch1 = cv.resize(patch, dsize=(0,0), fx=5, fy=5, interpolation=cv.INTER_NEAREST)
patch2 = cv.resize(patch, dsize=(0,0), fx=5, fy=5, interpolation=cv.INTER_LINEAR)
patch3 = cv.resize(patch, dsize=(0,0), fx=5, fy=5, interpolation=cv.INTER_CUBIC)

cv.imshow("Original", img)
cv.imshow("Resize Nearrest", patch1)
cv.imshow("Resize Linear", patch2)
cv.imshow("Resize Cubic", patch3)

cv.waitKey(0)
cv.destroyAllWindows()
