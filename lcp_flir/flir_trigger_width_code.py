import PySpin

serial_number = '20130136'
system = PySpin.System.GetInstance()
cam_list = system.GetCameras()
num_cameras = cam_list.GetSize()
print(f'found {num_cameras} cameras')
for i,cam in enumerate(cam_list):
    sn = cam.TLDevice.DeviceSerialNumber.GetValue()
    print(f'sn = {sn}')
    if serial_number == sn:
        break
cam_list.Clear()
cam.Init()
try:
    cam.AcquisitionStop()
    print ("Acquisition stopped")
except PySpin.SpinnakerException as ex:
    print("Acquisition was already stopped")
cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
cam.UserSetLoad()

cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono12Packed)

cam.ExposureMode.SetValue(PySpin.ExposureMode_TriggerWidth)
cam.AcquisitionMode.SetIntValue(PySpin.AcquisitionMode_Continuous)
print('setting frame rate enable to False')
cam.AcquisitionFrameRateEnable.SetValue(False)

print('setting Look Up Table Enable to False')
cam.LUTEnable.SetValue(False)

print('setting Gamma Enable to False')
cam.GammaEnable.SetValue(False)

print('setting TriggerSource Line0')
cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
#print('setting TriggerMode On')
# cam.TriggerMode.SetIntValue(PySpin.TriggerMode_On)
# print('setting TriggerSelector FrameStart')
cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
print('setting TriggerActivation RisingEdge')
cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)

cam.BeginAcquisition()

def get_image():
    from numpy import right_shift
    image_result = cam.GetNextImage()
    image_data = image_result.Convert(PixelFormat_Mono16).GetNDArray()
    image_data = right_shift(image_data,4)
    image_result.Release()
    return image_result

def clear_buffer():
    for i in range(100):
        temp = get_image()

def measure_once():
    clear_buffer();
    img = get_image();
    plt.figure();
    plt.imshow(img[1010:1040,1290:1320]);
    spot = img[1025:1028,1305:1308];
    print(spot.max(),spot.sum())
