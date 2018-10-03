"""
Multi Layer Perceptron in python using minpy. Gradient descent with noise

Network class stores 1D weight array with all parameters and biases of each layer
The network has a weight parser to handle the weights
Each layer has a parser to hangle parameters and biases

@author Adriana Perez Rotondo
@date Sept 2018
"""
import minpy
import minpy.numpy as np
#import autograd.numpy as np
from minpy.core import grad
#from autograd import grad
import matplotlib.pyplot as plt
from minpy.context import set_context, gpu
#from minpy.visualization.writer import LegacySummaryWriter as SummaryWriter
#import minpy.visualization.summaryOps as summaryOps
import datetime

summaries_dir = '/private/tmp/MLP_log'
NUM_INPUT = 10
NUM_OUTPUT = 10
NUM_LAYERS = 3
MAX_UNITS = 10
MIN_UNITS = 5
STUDENT_SCALING = [1,5,10]
PRINT_BIN = 100
GRAPH_OUTPUT_PATH = "./Figures/"
PARAMS_NAMES = 'params'
BIASES_NAMES = 'biases'
GAMMA1_FACTORS = [0.96, 0.04]

# ================================Utilities==================================
#    Parser Class to handle weights of layers and equalise_network_weights
# ===========================================================================

class WeightsParser(object):
    """A helper class to index into a parameter vector. """
    def __init__(self):
        self.idxs_and_shapes = {}   # dictionary with indexes for each parameter
        self.N = 0                  # total size

    def add_weights(self, name, shape):
        """Add to dictionary a tuple with the indeces of the parameter and shape
        shape should be a tuple with one or two elements (2,3) or (2,)
        """
        start = self.N
        # find the size
        if len(shape)==2:
            size = shape[0]*shape[1]
        elif len(shape)==1:
            size = shape[0]
        self.N += size  # update size
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)

    def get(self, vect, name):
        """Return the values of the variable with name in vect
        with original shape"""
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape) # reshape to original shape

    def set(self,vect,name,new):
        """Set the value of name in vect as new and return new array
        """
        idxs, shape = self.idxs_and_shapes[name]
        vect[idxs] = np.reshape(new,-1)
        return vect



# ================================Layer Class===============================
#        Layer class to handle weights and forward pass
# ===========================================================================

class full_layer(object):
    """
    layer class builds weights and biases of the layer and does forward pass
    """
    def __init__(self, size):
        self.size = size
        self.paramName = PARAMS_NAMES
        self.biasName  = BIASES_NAMES

    def build_weights_dict(self, input_shape):
        ''' builds a parser to store weights and biases of the layer
        returns the number of parameters of the layer and shape of layer
        '''
        # Input shape should be an int
        #input_size = np.prod(input_shape, dtype=int)
        #input_size = np.prod(input_shape)
        input_size = input_shape
        self.parser = WeightsParser()  # parser to store weights and biases
        # add to parser parameters and biases with specified shape
        self.parser.add_weights(self.paramName, (input_size, self.size))
        self.parser.add_weights(self.biasName, (self.size,))
        # return number of parameters of layer and shape
        return self.parser.N, (self.size,)

    def forward_pass(self, inputs, param_vector):
        """Get output of layer for inputs and param vector"""
        # get parameters and biases from vector with parser
        params = self.parser.get(param_vector, self.paramName)
        biases = self.parser.get(param_vector, self.biasName)
        # if inputs.ndim > 2:
        #     #inputs = inputs.reshape((inputs.shape[0], np.prod(inputs.shape[1:])))
        #     inputs = inputs.reshape((inputs.shape[0], inputs.shape[1]))
        # perform layer operation and return result
        return self.nonlinearity(np.dot(inputs[:, :], params) + biases)

class tanh_layer(full_layer):
    """layer with tanh nonlinearity"""
    def nonlinearity(self, x):
        return np.tanh(x)

class sigmoid_layer(full_layer):
    """layer with sigmoid nonlinearity"""
    def nonlinearity(self, x):
        return 1 / (1 + np.exp(-x))

class reLU_layer(full_layer):
    """layer with reLu nonlinearity"""
    def nonlinearity(self,x):
        return x*(x>0)


# ================================Network Classes===============================
#        Network classes: multi layer perceptron
# ==============================================================================

class MLP(object):
    ''' Multi-Layer perceptron'''
    def __init__(self, shape,
     activationFunctions = None,
     learningRate = None):
        self.shape =  shape                 # tuple with number of units in each layer
        self.numLayers = shape.shape[0]-1       # number of layers (not input)
        print(self.numLayers)
        self.layers = {}                    # dictionary of layers
        # number of synapses is size of network
        self.size = sum([shape[i-1]*shape[i] for i in range(1,shape.shape[0])])
        self.weight_init_var = self.size    # var of weight initialization
        self.parser = WeightsParser()       # weight's parser

        # learning rate
        if learningRate is None:
            self.lr = 0.02 * (self.size**(-0.5))    #default
        else:
            self.lr = learningRate

        # dictionary with the activation functions for each layer
        if activationFunctions and len(activationFunctions)==1:
            # if only one activation is given, set that as the activation
            # for all layers
            self.activationFunctions = [activationFunctions[0]\
             for i in range(self.numLayers)]
        else:
            self.activationFunctions = activationFunctions

        self.build()         # build layers, intialize weights, and functions

    def buildLayers(self):
        """Build layers of the network"""
        for l in range(1,self.numLayers+1):
            if not self.activationFunctions:
                # default activation function is sigmoid
                layer = sigmoid_layer(self.shape[l])
            elif self.activationFunctions[l] == "relu":
                layer = reLU_layer(self.shape[l])
            elif self.activationFunctions[l] == "sigmoid":
                layer = sigmoid_layer(self.shape[l])
            elif self.activationFunctions[l] == "tanh":
                layer = tanh_layer(self.shape[l])
            else: # default is sigmoid
                layer = sigmoid_layer(self.shape[l])
            # alocate space and shape of weights and biases in layer weight parser
            input_shape = self.shape[l-1]
            numWeights, _ = layer.build_weights_dict(input_shape)
            # add shape of layer to weight parser
            self.parser.add_weights(layer,(numWeights,))
            self.layers[l] = layer
        self.initializeWeights()

    def getLayerWeights(self,l,weightName):
        """Return parameters or biases (weightName) of layer l"""
        layer = self.layers[l]                          # layer
        weights = self.parser.get(self.weights,layer)   # weigths of layer
        return layer.parser.get(weights,weightName)     # get param/bias of layer

    def setLayerWeights(self,l,weightName,new):
        """Set param or weights of layer l to new"""
        layer = self.layers[l]                               # layer
        layerWeights = self.parser.get(self.weights,layer)   # weights of layer
        newW = layer.parser.set(layerWeights,weightName,new) # set in weight layer
        return self.parser.set(self.weights,layer,newW)      # set in net weights

    def forward(self,X):
        """Get output of network for input X"""
        current = X
        for l, layer in self.layers.items():
            w = self.parser.get(self.weights,layer)     # get layer weights
            current = layer.forward_pass(current, w)    # output of layer
        return current

    def _loss(self,predict,target):
        """Mean sqare error between target and predict"""
        #return 0.5*np.sum((predict - target)**2)/self.shape[-1]
        return 0.5*np.sum((predict - target)**2)

    def loss_f(self, input, target):
        """Loss from input and target"""
        predict = self.forward(input)
        return self._loss(predict,target)

    def loss_weights(self, w, input, target):
        """method to get loss for given weights (necessary for gradient)"""
        self.weights = w
        return self.loss_f(input,target)

    def build(self):
        """build network"""
        self.buildLayers()
        self.loss = self.loss_f
        self.gradient = grad(self.loss_weights,[0]) # diff with respect to weights

    def initializeWeights(self):
        """initialise weights"""
        # 12 is to normalise stdev or uniform random distribution
        self.weights = (12/self.weight_init_var)**(-0.5)*\
        (np.random.random_sample(self.parser.N)-1/2)

    def trainOneEpoch(self,input,target):
        """Train network for given input and target with gradient descent"""
        w = self.weights
        loss = self.loss(input,target)
        dw   = self.gradient(self.weights,input,target)
        # gradient descent
        w -= self.lr * dw
        deltaW = np.sum((self.weights -w)**2)   # change in weights
        self.weights = w                        # update weights
        return loss, deltaW

    def train(self,input, target, epochs):
        """Train network for all epochs"""
        #train_writer = SummaryWriter(summaries_dir + '/train')
        lossArray  = np.empty(epochs)   # array to store loss each epoch
        deltaW_vec = np.empty(epochs)   # array to store change in weights each epoch
        for i in range(epochs):
            loss, deltaW  = self.trainOneEpoch(input,target)
            lossArray[i]  = loss
            deltaW_vec[i] = deltaW
            if i % PRINT_BIN == 0:
                print('Iter {}, training loss {}'.format(i, loss))
            #summary1 = summaryOps.scalarSummary('loss', loss)
            #train_writer.add_summary(summary1, i)
        #train_writer.close()
        self.deltaW_vec = deltaW_vec
        self.loss_vec   = lossArray
        return lossArray

class NoisyMLP(MLP):
    def __init__(self, shape,
     activationFunctions = None,
     gamma = None,
     gamma1Factors = GAMMA1_FACTORS,
     delta_t = 1):
        MLP.__init__(self,shape,activationFunctions)
        self.gamma = gamma
        self.gamma1Factors = gamma1Factors
        self.delta_t = delta_t
        self.oldNoise = None

    def _makeNoise(self,shape):
        return np.random.standard_normal(shape)

    def _makeSystNoise(self,shape):
        newNoise = self._makeNoise(shape)
        newNoise = self.normalize(newNoise)
        newNoise *= self.gamma1Factors[0]
        oldNoise  = self.oldNoise
        if oldNoise is None:
            oldNoise = self._makeNoise(shape)
        oldNoise = self.normalize(oldNoise)
        oldNoise *= self.gamma1Factors[1]
        return newNoise + oldNoise

    def _getNorm(self,vect):
        return np.norm(vect)

    def normalize(self,vect):
        if np.norm(vect) is 0:
            return 0*vect
        else:
            return vect/np.norm(vect)

    def trainOneEpoch(self,input,target):
        w    = self.weights                             # weights
        loss = self.loss(input,target)                  # loss
        dw   = self.gradient(self.weights,input,target) # get gradient
        grad = dw
        noise1 = self._makeSystNoise(grad.shape)        # systematic noise
        noise2 = self._makeNoise(grad.shape)            # intrinsic noise
        # normalise and multiply by factor
        noise1 = self.normalize(noise1)
        noise2 = self.normalize(noise2)
        grad   = self.normalize(grad)
        noise1 *= -self.gamma[1]
        noise2 *= (-self.gamma[2])*(self.size/self.delta_t)**(-0.5)              # intrinsic noise scaling
        grad   *= self.gamma[0]
        newGrad = self.delta_t*(grad+noise1+noise2)
        # update weights
        w -= newGrad
        deltaW = np.sum((self.weights-w)**2)
        self.weights  = w
        self.oldNoise = noise1
        return loss, deltaW

class StudentsTeacher(object):
    def __init__(self,teacherSize,
    studentsSize = None,
    activationFunctions = None,
    gamma = [0.02,1,0],
    gamma1Factors = GAMMA1_FACTORS,
    delta_t = 1,
    numExamples = 1000,
    numEpochs = 1000):
        self.numEpochs           = numEpochs
        self.gamma               = gamma
        self.gamma1Factors       = gamma1Factors
        self.delta_t             = delta_t
        if studentsSize is None:
            studentsSize = [teacherSize]
        self.sizeStudents        = studentsSize
        self.activationFunctions = activationFunctions

        self.numStudents         = len(self.sizeStudents)
        self.students            = {}
        self.sizeNNs             = {}

        self.trainX       = np.random.randn(numExamples,teacherSize[0])
        self.trainingLoss = {}

        self._buildTeacher(teacherSize)
        self.buildStudents(teacherSize)

    def _buildNN(self,shape):
        """Method to build a neural network"""
        return NoisyMLP(shape, self.activationFunctions,\
        self.gamma, self.gamma1Factors, self.delta_t)

    def _buildTeacher(self,shape):
        """Build teacher"""
        self.teacher = self._buildNN(shape)
        self.teacherWeights = self.teacher.weights

    def buildStudents(self,sizeTeacher):
        """Build all students and set their weights to have same initial I-O"""
        for i in range(self.numStudents):
            size = self.sizeStudents[i]
            if size[0] != sizeTeacher[0] or size[-1] != sizeTeacher[-1]:
                print("Students must have same number of input anf output neurons as teacher")
                return
            student = self._buildNN(size)
            self.students[i] = student
            self.sizeNNs[i] = student.size
        self.setStudentsWeights()

    def _setStudLayerWeights(self,l,refParam,student,weightName):
        """Set param/biases of layer l for student to match refParam"""
        refShape = refParam.shape
        studentP = student.getLayerWeights(l,weightName)
        newP = np.zeros_like(studentP)
        if len(refShape)>1:
            newP[:refShape[0],:refShape[1]] = refParam
        else:
            newP[:refShape[0]] = refParam
        student.setLayerWeights(l,weightName, newP)

    def setStudentsWeights(self):
        """Set weights of students to match student 0 I-O"""
        refStudent  = self.students[0]
        refShape    = self.sizeStudents[0]
        for l in range(1,len(refShape)):
            refParam = refStudent.getLayerWeights(l,PARAMS_NAMES)
            refBiases = refStudent.getLayerWeights(l,BIASES_NAMES)
            for s in range(1,self.numStudents):
                student = self.students[s]
                self._setStudLayerWeights(l,refParam,student,PARAMS_NAMES)
                self._setStudLayerWeights(l,refBiases,student,BIASES_NAMES)

    def run(self):
        """Run student-teacher training return loss and size of students"""
        trainY = self.teacher.forward(self.trainX)
        for i in range(self.numStudents):
            self.trainingLoss[i] = self.students[i].train(self.trainX,trainY,self.numEpochs)
        return self.trainingLoss, self.sizeNNs

def teacherSize(numLayers = NUM_LAYERS, numInput = NUM_INPUT, numOutput = NUM_OUTPUT):
    size = np.random.randint(MIN_UNITS,MAX_UNITS,size = numLayers)
    size[0] = numInput
    size[-1] = numOutput
    return size

def studentSize(teacherSize, studentScaling = STUDENT_SCALING, scaleLayers = False):
    # creates the arrays of sizes for students from size  of teacher network scaled
    # number of layers for each student is multiplied by scaling
    if scaleLayers:
        layers = [int(x*teacherSize.shape) for x in studentScaling]
    else:
        layers = [int(1*teacherSize.size) for x in studentScaling]
    sizes = []
    print(layers)
    # for each student
    for i in range(len(layers)):
        # if the scaling is one return the same size as teacher
        if studentScaling[i] == 1:
            sizes.append(teacherSize)
        else:
            # scale max number of units added to each layer
            maxU = int(MAX_UNITS*studentScaling[i])
            # add random number of units to the layers in teacher
            newSize = teacherSize+np.random.randint(1,maxU, size=teacherSize.shape)
            print(newSize)
            if scaleLayers:
                # add layers with random number of units to the expanded layers of teacher
                newSize = np.concatenate((newSize,np.random.randint(MIN_UNITS,maxU,\
                size=(layers[i]-teacherSize.shape[0]))))
            sizes.append(newSize)
            # change number inputs and outputs to same as teacher
            sizes[i][0] = teacherSize[0]
            sizes[i][-1] = teacherSize[-1]
    return sizes

def studentSize_simple(teacher,sizes):
    return [[teacher[0],x,teacher[-1]] for x in sizes]

def plot_SSError_per_size(teacher,sizes,simulations=12):
    ssError = np.empty(shape=(simulations,len(sizes)))
    students = studentSize_simple(teacher,sizes)
    for sim in range(simulations):
        #tf.reset_default_graph()
        print(students)
        model = NoisyStudentsTeacher(teacher,students,epochs=1000)
        model.run()
        ssError[sim,:] = model.steadyState()
    print(ssError)
    meanError = ssError.mean(0)
    outputName = '{path}{date:%Y-%m-%d_%H%M%S_Size}.png'.format(path = GRAPH_OUTPUT_PATH, date=datetime.datetime.now())
    fig, ax = plt.subplots()
    sizesNN =[model.sizeNNs[i] for i in range(len(sizes))]
    ax.plot(sizesNN,meanError,'o')
    ax.set_ylabel('average Error')
    ax.set_xlabel('Number of synapses')
    fig.savefig(outputName)
    plt.close()

if __name__=='__main__':
    gamma = [0.05,1,0]
    gamma1Factors = [0.96,0.04]
    delta_t = 1
    numExamples = 1000
    numEpochs   = 1000
    teacher = teacherSize()
    print(teacher)
    students = studentSize(teacher)
    print(students)
    #teacher = [10,5,10]
    #students = [[10,5,10],[10,40,10],[10,400,10]]
    #teacher = (2,5,1)
    #students = [(2,5,1),(2,8,1),(2,10,1)]
    model = StudentsTeacher(teacher,students,'sigmoid',gamma,gamma1Factors,delta_t,numExamples,numEpochs)
    loss, sizes = model.run()
    outputName = '{path}{date:%Y-%m-%d_%H%M%S}_error.png'.format(path = GRAPH_OUTPUT_PATH, date=datetime.datetime.now())
    fig, ax = plt.subplots()
    for index, vec in loss.items():
        ax.plot(vec.asnumpy(), label = '{} synapses'.format(sizes[index]))
    ax.legend()
    ax.set_ylabel('Error')
    ax.set_xlabel('Epochs')
    ax.set_title('Error students of shapes {0} \nwith gamma {1}, T = {2}, {3} epochs and {4} examples'.format(str(students),str(gamma),delta_t,numEpochs,numExamples))
    plt.tight_layout()
    fig.savefig(outputName)
    plt.close()
    # shape = (10,5,10)
    # nn = NoisyMLP(shape,'sigmoid',gamma,delta_t)
    # #nn = MLP(shape,'sigmoid',gamma[0])
    # input_test = np.random.randn(numExamples,shape[0])
    # target_test = np.random.randn(numExamples,shape[-1])
    # err_vec = nn.train(input_test,target_test,numEpochs)
    # err_vec = np.zeros(numEpochs)
    # W = nn.weights
    # for i in range(numEpochs):
    #     err_vec[i] = nn.loss(input_test, target_test)
    #     g = nn.gradient(nn.weights,input_test,target_test)
    #     W -= learning_rate*g[0]
    #     nn.weights = W
    # print('loss', err_vec[-1])
    # print(err_vec)
    # outputName = '{path}{date:%Y-%m-%d_%H%M%S}_error.png'.format(path = GRAPH_OUTPUT_PATH, date=datetime.datetime.now())
    # fig, ax = plt.subplots()
    # ax.plot(err_vec.asnumpy())
    # #ax.plot(nn.deltaW_vec.asnumpy())
    # ax.set_ylabel('Error')
    # ax.set_xlabel('Epochs')
    # ax.set_title('Error for MLP of shape {0} \nwith gamma {1}, T = {2}, {3} epochs and {4} examples'.format(str(shape),str(gamma),delta_t,numEpochs,numExamples))
    # plt.tight_layout()
    # fig.savefig(outputName)
    # plt.close()
    # outputName = '{path}{date:%Y-%m-%d_%H%M%S}_weights.png'.format(path = GRAPH_OUTPUT_PATH, date=datetime.datetime.now())
    # fig, ax = plt.subplots()
    # ax.plot(nn.deltaW_vec.asnumpy())
    # ax.set_ylabel('Change in weights')
    # ax.set_xlabel('Epochs')
    # ax.set_title('Change in weights for MLP of shape {0} \nwith gamma {1}, T = {2}, {3} epochs and {4} examples'.format(str(shape),str(gamma),delta_t,numEpochs,numExamples))
    # plt.tight_layout()
    # fig.savefig(outputName)
    # plt.close()
