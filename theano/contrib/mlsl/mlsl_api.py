from ctypes import c_void_p
import mlsl
import theano

def addr(x):
    xaddr, offset = x.ctypes.data_as(c_void_p), 0
    for i in range(len(x.shape)):
        if x.strides[i] < 0: offset += (x.shape[i]-1)*x.strides[i]
    xaddr.value += offset
    return xaddr

# For data parallelism
class AllReduce(theano.Op):
    def __init__(self,mlsl_obj, dist, count):
        self.dist = dist
        self.count = count
        self.mlsl_obj = mlsl_obj

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        outputs = [x.type(),]
        return theano.Apply(self,[x],outputs)

    def perform(self,node,inputs,outputs):
        # In place
        dist = self.dist
        x, = inputs
        y, = outputs
        #print 'in perform ',x.shape
        y[0]=x
        req=dist.all_reduce(send_buf=addr(x), recv_buf=addr(x), count=self.count, data_type=0, red_type=0, group_type=2)
        self.mlsl_obj.wait(req)

    def grad(self,inputs,grads):
        return [theano.gradient.DisconnectedType()(),]