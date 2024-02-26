
node.warn ("Starting manager script node")
import marshal
from math import sin, cos, atan2, pi, hypot, degrees, floor
left_shoulder = 11
right_shoulder = 12
left_hip = 23
right_hip = 24
class BufferMgr:
    def __init__(self):
        self._bufs = {}
    def __call__(self, size):
        try:
            buf = self._bufs[size]
        except KeyError:
            buf = self._bufs[size] = Buffer(size)
            node.warn (f"New buffer allocated: {size}")
        return buf
buffer_mgr = BufferMgr()
def send_result(type, lm_score=0, rect_center_x=0, rect_center_y=0, rect_size=0, rotation=0, lms=0, lms_world=0, xyz_ref=0, xyz=0, xyz_zone=0):
    result = dict([("type", type), ("lm_score", lm_score), ("rotation", rotation),
            ("rect_center_x", rect_center_x), ("rect_center_y", rect_center_y), ("rect_size", rect_size), 
            ("lms", lms), ('lms_world', lms_world),
            ("xyz_ref", xyz_ref), ("xyz", xyz), ("xyz_zone", xyz_zone)])
    result_serial = marshal.dumps(result, 2)
    buffer = buffer_mgr(len(result_serial))  
    buffer.getData()[:] = result_serial
    node.warn ("len result:"+str(len(result_serial)))
    node.io['host'].send(buffer)
    node.warn ("Manager sent result to host")
def rr2img(rrn_x, rrn_y): 
    X = sqn_rr_center_x + sqn_rr_size * ((rrn_x - 0.5) * cos_rot + (0.5 - rrn_y) * sin_rot)
    Y = sqn_rr_center_y + sqn_rr_size * ((rrn_y - 0.5) * cos_rot + (rrn_x - 0.5) * sin_rot)
    return X, Y
norm_pad_size = 252 / 1152
def is_visible(lm_id):
    return lms[lm_id*5+3] > 0.5 
def is_in_image(sqn_x, sqn_y):
    return  norm_pad_size < sqn_y < 1 - norm_pad_size
send_new_frame_to_branch = 1
next_roi_lm_idx = 33*5
cfg_pre_pd = ImageManipConfig()
cfg_pre_pd.setResizeThumbnail(224, 224, 0, 0, 0)
while True:
    if send_new_frame_to_branch == 1: 
        node.io['pre_pd_manip_cfg'].send(cfg_pre_pd)
        node.warn ("Manager sent thumbnail config to pre_pd manip")
        detection = node.io['from_post_pd_nn'].get().getLayerFp16("result")
        node.warn ("Manager received pd result: "+str(detection))
        pd_score, sqn_rr_center_x, sqn_rr_center_y, sqn_scale_x, sqn_scale_y = detection
        if pd_score < 0.5:
            send_result(0)
            continue
        scale_center_x = sqn_scale_x - sqn_rr_center_x
        scale_center_y = sqn_scale_y - sqn_rr_center_y
        sqn_rr_size = 2 * 1.25 * hypot(scale_center_x, scale_center_y)
        rotation = 0.5 * pi - atan2(-scale_center_y, scale_center_x)
        rotation = rotation - 2 * pi *floor((rotation + pi) / (2 * pi))
    sin_rot = sin(rotation) 
    cos_rot = cos(rotation)
    rr = RotatedRect()
    rr.center.x    = sqn_rr_center_x
    rr.center.y    = (sqn_rr_center_y * 1152 - 252) / 648
    rr.size.width  = sqn_rr_size
    rr.size.height = sqn_rr_size * 1152 / 648
    rr.angle       = degrees(rotation)
    cfg = ImageManipConfig()
    cfg.setCropRotatedRect(rr, True)
    cfg.setResize(256, 256)
    node.io['pre_lm_manip_cfg'].send(cfg)
    node.warn ("Manager sent config to pre_lm manip")
    lm_result = node.io['from_lm_nn'].get()
    node.warn ("Manager received result from lm nn")
    lm_score = lm_result.getLayerFp16("Identity_1")[0]
    if lm_score > 0.7:
        lms = lm_result.getLayerFp16("Identity")
        lms_world = lm_result.getLayerFp16("Identity_4")[:99]
        xyz = 0
        xyz_zone = 0
        xyz_ref = 0
        send_result(send_new_frame_to_branch, lm_score, sqn_rr_center_x, sqn_rr_center_y, sqn_rr_size, rotation, lms, lms_world, xyz_ref, xyz, xyz_zone)
        if not False:
            send_new_frame_to_branch = 2 
            rrn_rr_center_x = lms[next_roi_lm_idx] / 256
            rrn_rr_center_y = lms[next_roi_lm_idx+1] / 256
            rrn_scale_x = lms[next_roi_lm_idx+5] / 256
            rrn_scale_y = lms[next_roi_lm_idx+6] / 256
            sqn_scale_x, sqn_scale_y = rr2img(rrn_scale_x, rrn_scale_y)
            sqn_rr_center_x, sqn_rr_center_y = rr2img(rrn_rr_center_x, rrn_rr_center_y)
            scale_center_x = sqn_scale_x - sqn_rr_center_x
            scale_center_y = sqn_scale_y - sqn_rr_center_y
            sqn_rr_size = 2 * 1.25 * hypot(scale_center_x, scale_center_y) 
            rotation = 0.5 * pi - atan2(-scale_center_y, scale_center_x)
            rotation = rotation - 2 * pi *floor((rotation + pi) / (2 * pi))    
    else:
        send_result(send_new_frame_to_branch, lm_score)
        send_new_frame_to_branch = 1
