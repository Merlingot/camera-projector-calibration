"""
Fonctions pour controler une camera manta de Allied Vision
"""

import pymba
import cv2
from skimage.io import imsave, imshow
from skimage.viewer import ImageViewer
import numpy as np
from pymba import Vimba, VimbaException
from pymba import Frame


def test_pymba():
    with Vimba() as vimba:
        camera = vimba.camera(0)
        camera.open()

        camera.arm('SingleFrame')

        # capture a single frame, more than once if desired
        for i in range(1):
            try:
                frame = camera.acquire_frame()
                display_frame(frame, 0)
            except VimbaException as e:
                # rearm camera upon frame timeout
                if e.error_code == VimbaException.ERR_TIMEOUT:
                    print(e)
                    camera.disarm()
                    camera.arm('SingleFrame')
                else:
                    raise

        camera.disarm()
        camera.close()


def test():

    with Vimba() as vimba:

        # Find and open camera
        cam=vimba.camera(0)
        cam.open()

        # Test single frame acquisition
        cam.arm('SingleFrame')
        frame=take_frame(cam)
        display_frame(frame)
        save_frame(frame, 'test_manta')

        # Test continous frame acquisition
        cam.arm('Continuous', display_frame)
        cam.start_frame_acquisition()
        input('Press enter to stop streaming')
        cam.stop_frame_acquisition()

        # Close camera
        cam.disarm()
        cam.close()



def capture_series(name, destination=None):
    """
    Capturer une série de X photos avec confirmation de l'usager avant chaque capture.

    Prend les X photos avec la caméra détectée et les enregistre sous le nom <image_name_i>.png en format png dans le répertoire <destination>.
    name (str):
        nom pour enregistrer l'image sans l'extension
    destination (str):
        path relatif/absolu au répertoire de sauvegarge
        default=None. Si non spécifié, sauvegarde dans le repertoire actuel.
    """
    i=0
    continue_callibration=True
    with Vimba() as vimba:
        cam=vimba.camera(0)
        cam.open()
        input('Ready fo callibration :)')
        while continue_callibration == True:
            # arm the camera and provide a function to be called upon frame
            cam.arm('Continuous', display_frame)
            cam.start_frame_acquisition() # stream images until input
            user_input = input('Press \n (y) to take image \n (q) to exit')
            if user_input == 'y' :
                cam.stop_frame_acquisition()
                cam.arm('SingleFrame')
                print('Taking image...')
                frame=take_frame(cam)
                print('Saving image...')
                name_i=name+'_{}'.format(str(i))
                save_frame(frame, name_i, destination)
                i+=1
            elif user_input == 'q':
                cam.stop_frame_acquisition()
                print('Exiting...')
                cam.disarm()
                cam.close()
                continue_callibration=False
            else:
                cam.stop_frame_acquisition()
                print('Please enter a valid key (y/q)')


def get_camera():
    """ Trouver une camera compatible """
    with Vimba() as vimba:
        # provide camera index or id
        cam = vimba.camera(0)
        return cam

def take_frame(cam):
    """ Capture a single frame """
    for i in range(1):
        try:
            frame = cam.acquire_frame()
        except VimbaException as e:
            # rearm camera upon frame timeout
            if e.error_code == VimbaException.ERR_TIMEOUT:
                print(e)
                cam.disarm()
                cam.arm('SingleFrame')
            else:
                raise
    return frame

def display_frame(frame):
    """
    Displays the acquired frame.
        frame: The frame object to display.
    """
    image = frame_to_image(frame)
    imshow(image)
    #viewer = ImageViewer(image); viewer.show()

def save_frame(frame, name, destination=None):
    """
    Saves the frame as a png image.
        frame : The frame object to save
        name : The name of the image to save (without .png extension)
        destination (optional) : the absolute/relative path to the directory where to save the image
    """
    image = frame_to_image(frame)
    path = "{}{}.png".format( destination if destination is not None else "", name)
    isSaved = imsave(path, image, format='png')

def frame_to_image(frame):
    """
    Takes the data in the camera buffer and converts it to a readable uint8 numpy array
    """
    # get a copy of the frame data
    image = frame.buffer_data_numpy()
    #Convert the datatype to np.uint8
    new_image = image.astype(np.uint8)

    return new_image
