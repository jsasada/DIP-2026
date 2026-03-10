import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """

    warped_image = np.array(image)
    if (len(source_pts) <= 1) or (len(target_pts) <= 1):
        return warped_image

    p = np.array(target_pts)
    q = np.array(source_pts)

    M = image.shape[1]
    N = image.shape[0]
    grid_x, grid_y = np.meshgrid(np.arange(M), np.arange(N)) # np.meshgrid(np.linspace(0, image.shape[1]-1, M), np.linspace(0, image.shape[0]-1, N))
    v = np.vstack((grid_x.flatten(), grid_y.flatten()))
    v = np.transpose(v)

    diff = v[:, None, :] - p[None, :, :]
    diff = np.sum(np.pow(diff, 2), axis=2) + eps
    w = np.pow(diff, -alpha)
    w_sum = np.sum(w, axis=1)

    p_ast = np.matmul(w, p) / w_sum[:, None]
    q_ast = np.matmul(w, q) / w_sum[:, None]

    p_hat = p[None, :, :] - p_ast[:, None, :]
    q_hat = q[None, :, :] - q_ast[:, None, :]

    p_hat_T_q_hat = np.multiply(p_hat[:, :, :, None], q_hat[:, :, None, :])
    p_hat_T_p_hat = np.multiply(p_hat[:, :, :, None], p_hat[:, :, None, :])
    M_1 = np.sum(np.multiply(p_hat_T_q_hat, w[:, :, None, None]), axis=1)
    M_2 = np.sum(np.multiply(p_hat_T_p_hat, w[:, :, None, None]), axis=1)
    aff_M = np.linalg.pinv(M_2) @ M_1

    v_new = np.squeeze(np.matmul((v - p_ast)[:, None, :], aff_M), axis=1) + q_ast

    dst = v_new.reshape(N, M, 2)
    map_x = dst[:, :, 0].astype(np.float32)
    map_y = dst[:, :, 1].astype(np.float32)

    warped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
