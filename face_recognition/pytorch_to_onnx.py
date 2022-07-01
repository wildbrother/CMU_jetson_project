import torch
import torchvision.models as models

from facenet_pytorch import MTCNN, InceptionResnetV1

# mtcnn : (225, 225, 3)
# net : torch.Size([1, 3, 160, 160])


device = torch.device("cuda:0")

# net = InceptionResnetV1(pretrained='vggface2')
# net.eval()
# print('net is ready')
# net = net.to(device)

# input_size = (1, 3, 160, 160)
# output_onnx = 'InceptionResnetV1.onnx'
# input_names = ["input_0"]
# output_names = ["output_0"]

# # Let's create a dummy input tensor  
# dummy_input = torch.randn(input_size, requires_grad=True).to(device)

# # Export the model   
# torch.onnx.export(net,         # model being run 
#         dummy_input,       # model input (or a tuple for multiple inputs) 
#         output_onnx,       # where to save the model  
#         export_params=True,  # store the trained parameter weights inside the model file 
#         opset_version=10,    # the ONNX version to export the model to 
#         do_constant_folding=True,  # whether to execute constant folding for optimization 
#         input_names = ['modelInput'],   # the model's input names 
#         output_names = ['modelOutput'], # the model's output names 
#         dynamic_axes={'modelInput' : {0 : 'batch_size'}, 'modelOutput' : {0 : 'batch_size'}}) 
# print(" ") 
# print('Model has been converted to ONNX')



mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.9, post_process=True,
    device=device, keep_all=True
)


mtcnn.eval()
print('mtcnn is ready')
mtcnn = mtcnn.to(device)

input_size = (225, 225, 3)
output_onnx = 'mtcnn.onnx'
input_names = ["input_0"]
output_names = ["output_0"]

dummy_input = torch.randn(input_size, requires_grad=True).to(device)

# Export the model   
torch.onnx.export(mtcnn,         # model being run 
        dummy_input,       # model input (or a tuple for multiple inputs) 
        output_onnx,       # where to save the model  
        export_params=True,  # store the trained parameter weights inside the model file 
        opset_version=10,    # the ONNX version to export the model to 
        do_constant_folding=True,  # whether to execute constant folding for optimization 
        input_names = ['modelInput'],   # the model's input names 
        output_names = ['modelOutput'], # the model's output names 
        dynamic_axes={'modelInput' : {0 : 'batch_size'}, 'modelOutput' : {0 : 'batch_size'}}) 
print(" ") 
print('Model has been converted to ONNX')