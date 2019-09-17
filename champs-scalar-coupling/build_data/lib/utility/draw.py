import os
#qt bug ???
os.environ['QT_XKB_CONFIG_ROOT']='/usr/share/X11/xkb/'

from lib.include import *
import matplotlib.cm


# draw -----------------------------------
def image_show(name, image, resize=1):
    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_GUI_NORMAL)  #WINDOW_NORMAL
    #cv2.namedWindow(name, cv2.WINDOW_GUI_EXPANDED)  #WINDOW_GUI_EXPANDED
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))


def image_show_norm(name, image, max=None, min=None, resize=1):
    if max is None: max=image.max()
    if min is None: min=image.min()

    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_GUI_NORMAL)  #WINDOW_NORMAL
    cv2.imshow(name, ((image-min)/(max-min)*255).astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))

def draw_shadow_text(img, text, pt,  fontScale, color, thickness, color1=None, thickness1=None):
    if color1 is None: color1=(0,0,0)
    if thickness1 is None: thickness1 = thickness+2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color1, thickness1, cv2.LINE_AA)
    cv2.putText(img, text, pt, font, fontScale, color,  thickness,  cv2.LINE_AA)

def to_color_image(image, max=None):
    if max is None: max=image.max()

    image = (image/max*255).astype(np.uint8)
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    return image



##http://stackoverflow.com/questions/26690932/opencv-rectangle-with-dotted-or-dashed-lines
def draw_dotted_line(image, pt1, pt2, color, thickness=1, gap=20):

    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if gap==1:
        for p in pts:
            cv2.circle(image,p,thickness,color,-1,cv2.LINE_AA)
    else:
        def pairwise(iterable):
            "s -> (s0, s1), (s2, s3), (s4, s5), ..."
            a = iter(iterable)
            return zip(a, a)

        for p, q in pairwise(pts):
            cv2.line(image,p, q, color,thickness,cv2.LINE_AA)


def draw_dotted_poly(image, pts, color, thickness=1, gap=20):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        draw_dotted_line(image,s,e,color,thickness,gap)


def draw_dotted_rect(image, pt1, pt2, color, thickness=1, gap=3):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])]
    draw_dotted_poly(image, pts, color, thickness, gap)

def draw_screen_rect(image, pt1, pt2, color, alpha=0.5):
    x1, y1 = pt1
    x2, y2 = pt2
    image[y1:y2,x1:x2,:] = (1-alpha)*image[y1:y2,x1:x2,:] + (alpha)*np.array(color, np.uint8)



# def draw_mask(image, mask, color=(255,255,255), α=1,  β=0.25, λ=0., threshold=32 ):
#     # image * α + mask * β + λ
#
#     if threshold is None:
#         mask = mask/255
#     else:
#         mask = clean_mask(mask,threshold,1)
#
#     mask  = np.dstack((color[0]*mask,color[1]*mask,color[2]*mask)).astype(np.uint8)
#     image[...] = cv2.addWeighted(image, α, mask, β, λ)
#


# def draw_contour(image, mask, color=(0,255,0), thickness=1, threshold=127):
#     ret, thresh = cv2.threshold(mask,threshold,255,0)
#     ret = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     hierarchy = ret[0]
#     contours  = ret[1]
#     #image[...]=image
#     cv2.drawContours(image, contours, -1, color, thickness, cv2.LINE_AA)
#     ## drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None): # real signature unknown; restored from __doc__
#
#


def to_color(s, color=None):

    if type(color) in [str] or color is None:
        #https://matplotlib.org/xkcd/examples/color/colormaps_reference.html

        if color is None: color='cool'
        color = matplotlib.get_cmap(color)(s)
        b = int(255*color[2])
        g = int(255*color[1])
        r = int(255*color[0])

    elif type(color) in [list,tuple]:
        b = int(s*color[0])
        g = int(s*color[1])
        r = int(s*color[2])

    return b,g,r



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    image = np.zeros((50,50,3), np.uint8)
    cv2.rectangle(image, (0,0),(49,49), (0,0,255),1) #inclusive

    image[8,8]=[255,255,255]

    image_show('image',image,10)
    cv2.waitKey(0)


    print('\nsucess!')