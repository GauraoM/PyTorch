#### Creating Tensors

Vector: t = torch.tensor([1.,2,3,4])
Matrix: t = torch.tensor([[5.,6],[7,8],[9,10]])

#### Tensor operations and gradient
required_grad = True ,helps us to get the derivative like dy/dw

w = torch.tensor(4. required_grad = True)

#### Converting numpy array to torch tensor

x = ([[1,2,3],[4,5,6]])
y = torch.from_numpy(x) or torch.tensor(x)

#### tensor to numpy array  
z = y.numpy()

#### Creating random matrices
torch.randn gives values between -1 to 1

w = torch.randn(2,3,requires_grad = True)

#### Matrix multiplication  and transpose
.t represents Transpose
@ represents multiplication

x@w.t()+b

#### addition and number of elements
addition: torch.sum(diff_sqr)
numel: no.of elements

torch.sum(diff)
diff.numel()

#### Tensor Dataset
It allows small section of training data using array indexing notation

train_ds = TensorDataset(inputs, targets)

#### DataLoader 
Can split the data into batches of a predefined size while training. It also provides other utilities like shuffling and random sampling of the data.

train_dl = DataLoader(train_ds,batch_size, shuffle=True)

#### creating weights and biases
model = nn.Linear(input,output)

model = nn.Linear(3,2)

#### optimizers
Following is an example of implementing Stochastic Gradient Descent, which is used as an optimizer.

opt = torch.optim.SGD(model.parameters(), lr=1e-5)

#### update parameters using gradient
All optimizers implement a step() method, that updates the parameters.
optimizer.step()

opt.step()

#### ToTensor
It converts the images to pytorch tensors

dataset = MNIST(root='data/', 
                train=True,
                transform=transforms.ToTensor())

#### To resize the shape 
view(): it ensure we are using same block of memory
reshape: Can also resize the shape

xb = xb.view(xb.size(0), -1)
Here the xb.shape(0) means the batch size as we want to convert it to 100*784. -1 represents the multiplication of dimension of images 1*28*28 = 784. 

#### Load the dataset as PyTorch Tensor
ImageFolder class from torchvision to load the dataset as pytorch tensor

dataset = ImageFolder(data_dir + "/train", transform=ToTensor())

#### Changing Dimension of an image(permute)
Move position 1--->0,2--->1,0--->2

plt.imshow(img.permute(1, 2, 0))

####  Data Transform (Normalization and data augmentation)
We have to import "import torchvision.transforms as tt" then provide the stats as mean values for subtraction from image tensor to normalize and divide by standard deviation values for each channel in the image. Compose is used to perform several transform together.

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode="reflect"),
                         tt.RandomHorizontalFlip(),
                         tt.ToTensor(),
                         tt.Normalize(*stats)
                         ])

#### Learning rate finder
FastAI allows to select you good learning rate by looking at the graph of loss vs learning rate.

learner = Learner(data, model, loss_func=F.cross_entropy, metrics=[accuracy])
learner.lr_find()

#### Return new Tensor(Detach)
Do not track any gradient,the result will never require gradient. Just return new Tensor.

gen_imgs = denorm(y.reshape((-1, 28,28)).detach())

#### Clamp(To produce the output within some range)
This can be used to produce the output within certain range as provided.As a min and max values are provided.

torch.clamp(0,1)

#### Saving object to local disk(torch.save)
torch.save used to save an object to a disk

torch.save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL_use_new_zipfile_serialization=True)

Ex:torch.save(G.state_dict(), 'G.ckpt') # save checkpoints


