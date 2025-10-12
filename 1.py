import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
 
main_img_path = "3.jpg"
flag_img_path = "image.jpg"

main_img = cv.imread(main_img_path)
flag_img = cv.imread(flag_img_path)
main_img_copy = main_img.copy()
img_points = []

#function for point selecting by mouse clicks
def select_points(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        img_points.append([x, y])
        print(f"Point selected: {x}, {y}")
        
        cv.circle(main_img, (x, y), 5, (0, 255, 0), -1)
        cv.imshow("Select 4 points", main_img)


cv.imshow("Select 4 points", main_img)
cv.setMouseCallback("Select 4 points", select_points)

while len(img_points) < 4:
    cv.waitKey(1)

cv.destroyAllWindows()

main_img_points = np.array(img_points, dtype =np.float32)
flag_points = np.array([
    [0, 0],
    [flag_img.shape[1], 0],
    [flag_img.shape[1], flag_img.shape[0]],
    [0, flag_img.shape[0]]
    ], dtype=np.float32)

H,_ = cv.findHomography(flag_points, main_img_points)
warped_flag = cv.warpPerspective(flag_img, H, (main_img.shape[1],main_img.shape[0]))

mask = np.zeros_like(main_img, dtype = np.uint8)
cv.fillConvexPoly(mask, main_img_points.astype(int), (255,255,255))

flag_region = cv.bitwise_and(warped_flag, mask)
background = cv.bitwise_and(main_img, cv.bitwise_not(mask))
blend_img = cv.add(background, flag_region)

fig, axes = plt.subplots(1, 3, figsize=(12, 6))

# Show original background image
axes[0].imshow(cv.cvtColor(main_img, cv.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[0].axis('off')

# show flag
axes[1].imshow(cv.cvtColor(flag_img, cv.COLOR_BGR2RGB))
axes[1].set_title("Flag image")
axes[1].axis('off')

# Show blended result
axes[2].imshow(cv.cvtColor(blend_img, cv.COLOR_BGR2RGB))
axes[2].set_title("Blended Result")
axes[2].axis('off')

plt.tight_layout()
plt.show()