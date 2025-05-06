import torch

def check_conv3d_prepack_support():
    """Specifically check if conv3d_prepack is supported in the current PyTorch version."""
    
    # Check available backends
    print("Available quantization backends:")
    print(torch.backends.quantized.supported_engines)
    
    # Look for specific conv3d_prepack operator
    quantized_ops = dir(torch.ops.quantized)
    prepack_ops = [op for op in quantized_ops if 'prepack' in op.lower()]
    print("\nAvailable prepack operators:")
    print(prepack_ops)
    
    # Check specifically for conv3d_prepack
    has_conv3d_prepack = 'conv3d_prepack' in quantized_ops
    print(f"\nconv3d_prepack operator available: {has_conv3d_prepack}")
    
    # Check for any conv3d operators
    conv3d_ops = [op for op in quantized_ops if 'conv3d' in op.lower()]
    print(f"\nAll conv3d related operators:")
    print(conv3d_ops)
    
    # Test conv3d_prepack functionality on CPU and CUDA
    print("\nTesting conv3d_prepack functionality:")
    try:
        # Create test tensors for CPU
        x = torch.randn(1, 3, 4, 4, 4)
        weight = torch.randn(6, 3, 3, 3, 3)
        bias = torch.randn(6)
        
        # Try to use the operator directly if available
        if hasattr(torch.ops.quantized, 'conv3d_prepack'):
            print("Found conv3d_prepack in torch.ops.quantized")
            packed_weight = torch.ops.quantized.conv3d_prepack(
                weight, 
                bias,
                [1, 1, 1],  # stride 
                [0, 0, 0],  # padding
                [1, 1, 1],  # dilation
                1           # groups
            )
            print("Successfully called conv3d_prepack on CPU")
        else:
            print("conv3d_prepack not found in torch.ops.quantized")
    
        # Try the higher-level API if available
        if hasattr(torch.nn.quantized, 'Conv3d'):
            print("\nTesting torch.nn.quantized.Conv3d:")
            conv3d = torch.nn.quantized.Conv3d(3, 6, kernel_size=3, padding=1)
            y = conv3d(torch.quantize_per_tensor(x, scale=0.1, zero_point=0, dtype=torch.quint8))
            print("Successfully executed quantized Conv3d on CPU")
        else:
            print("\ntorch.nn.quantized.Conv3d is not available")
    except Exception as e:
        print(f"Error testing Conv3d on CPU: {e}")
    
    # Check CUDA support for conv3d_prepack
    print("\nChecking CUDA support for conv3d_prepack:")
    try:
        if torch.cuda.is_available():
            x_cuda = x.cuda()
            weight_cuda = weight.cuda()
            bias_cuda = bias.cuda()
            
            try:
                if hasattr(torch.ops.quantized, 'conv3d_prepack'):
                    # Try to execute on CUDA
                    print("Attempting conv3d_prepack on CUDA tensors...")
                    packed_weight_cuda = torch.ops.quantized.conv3d_prepack(
                        weight_cuda, 
                        bias_cuda,
                        [1, 1, 1],  # stride 
                        [0, 0, 0],  # padding
                        [1, 1, 1],  # dilation
                        1           # groups
                    )
                    print("Successfully called conv3d_prepack on CUDA")
            except Exception as e:
                print(f"conv3d_prepack failed on CUDA: {e}")
                
            # Try to quantize and run on CUDA
            try:
                print("\nAttempting to quantize and run Conv3d on CUDA...")
                x_q = torch.quantize_per_tensor(x, scale=0.1, zero_point=0, dtype=torch.quint8)
                if hasattr(torch.nn.quantized, 'Conv3d'):
                    conv3d = torch.nn.quantized.Conv3d(3, 6, kernel_size=3, padding=1)
                    try:
                        conv3d.cuda()
                        print("Successfully moved quantized Conv3d to CUDA")
                    except Exception as e:
                        print(f"Failed to move quantized Conv3d to CUDA: {e}")
            except Exception as e:
                print(f"Failed to test quantized Conv3d on CUDA: {e}")
        else:
            print("CUDA not available")
    except Exception as e:
        print(f"Error testing CUDA support: {e}")
    
    # Check if there's a way to run quantized operations on CUDA
    print("\nChecking alternative approaches for CUDA quantization:")
    if torch.cuda.is_available():
        try:
            from torch.ao.quantization import QuantStub, DeQuantStub
            
            # Define a simple model with QuantStub and DeQuantStub
            class SimpleConv3dModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.quant = QuantStub()
                    self.conv = torch.nn.Conv3d(3, 6, kernel_size=3, padding=1)
                    self.dequant = DeQuantStub()
                
                def forward(self, x):
                    x = self.quant(x)
                    x = self.conv(x)
                    x = self.dequant(x)
                    return x
            
            # Create and prepare model
            model = SimpleConv3dModel().cuda()
            model.eval()
            
            # Configure quantization
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # Try to prepare and quantize
            try:
                print("Attempting QAT preparation on CUDA...")
                torch.quantization.prepare_qat(model, inplace=True)
                print("QAT preparation successful on CUDA")
            except Exception as e:
                print(f"QAT preparation failed on CUDA: {e}")
                
                # Try moving to CPU first
                print("\nTrying CPU quantization then moving to CUDA...")
                model = SimpleConv3dModel().cpu()
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare_qat(model, inplace=True)
                
                # Fake training
                model(x)
                
                # Convert and try moving to CUDA
                torch.quantization.convert(model, inplace=True)
                try:
                    model.cuda()
                    print("Successfully moved quantized model to CUDA")
                except Exception as e:
                    print(f"Failed to move quantized model to CUDA: {e}")
        except Exception as e:
            print(f"Error testing alternative approaches: {e}")

# Run the check
check_conv3d_prepack_support()