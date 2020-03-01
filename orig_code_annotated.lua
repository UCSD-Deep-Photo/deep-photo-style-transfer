'''
REMOVED ALL IMPORTS AND CONFIGS
'''

function setup_gpu(params)
  '''GPU MGMT'''
end

function setup_multi_gpu(net, params)
'''GPU MGMT'''
end

function build_filename(output_image, iteration)
'''BUILDS FILES'''
end


-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end


-- Undo the above preprocessing.
function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(256.0)
  return img
end


-- Combine the Y channel of the generated image and the UV channels of the
-- content image to perform color-independent style transfer.
function original_colors(content, generated)'''REMOVED FOR SIMPLICITY'''end


'''
CUSTOM CONTENT LOSS FUNCTION
'''
-- Define an nn Module to compute content loss in-place
local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength, normalize)
  'strength = content_weight: How much to weight the content reconstruction term. Default is 5e0.'
  'normalize: optional L1 normalization'
  parent.__init(self)
  self.strength = strength
  self.target = torch.Tensor()
  self.normalize = normalize or false
  self.loss = 0
  self.crit = nn.MSECriterion()
  self.mode = 'none'
end

function ContentLoss:updateOutput(input)
  if self.mode == 'loss' then
    self.loss = self.crit:forward(input, self.target) * self.strength
  elseif self.mode == 'capture' then
    self.target:resizeAs(input):copy(input)
  end
  self.output = input
  return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
  if self.mode == 'loss' then
    if input:nElement() == self.target:nElement() then
      self.gradInput = self.crit:backward(input, self.target)
    end
    if self.normalize then
      self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
    end
    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
  else
    self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  end
  return self.gradInput
end


local Gram, parent = torch.class('nn.GramMatrix', 'nn.Module')

function Gram:__init()
  parent.__init(self)
end

'''
COVARIANCE MATRIX
'''
function Gram:updateOutput(input)
  assert(input:dim() == 3)
  local C, H, W = input:size(1), input:size(2), input:size(3)
  local x_flat = input:view(C, H * W)
  self.output:resize(C, C)
  self.output:mm(x_flat, x_flat:t())
  return self.output
end

function Gram:updateGradInput(input, gradOutput)
  assert(input:dim() == 3 and input:size(1))
  local C, H, W = input:size(1), input:size(2), input:size(3)
  local x_flat = input:view(C, H * W)
  self.gradInput:resize(C, H * W):mm(gradOutput, x_flat)
  self.gradInput:addmm(gradOutput:t(), x_flat)
  self.gradInput = self.gradInput:view(C, H, W)
  return self.gradInput
end


-- Define an nn Module to compute style loss in-place
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(strength, normalize)
  parent.__init(self)
  self.normalize = normalize or false
  self.strength = strength
  self.target = torch.Tensor()
  self.mode = 'none'
  self.loss = 0

  self.gram = nn.GramMatrix()
  self.blend_weight = nil
  self.G = nil
  self.crit = nn.MSECriterion()
end

function StyleLoss:updateOutput(input)
  self.G = self.gram:forward(input)
  self.G:div(input:nElement())
  if self.mode == 'capture' then
    if self.blend_weight == nil then
      self.target:resizeAs(self.G):copy(self.G)
    elseif self.target:nElement() == 0 then
      self.target:resizeAs(self.G):copy(self.G):mul(self.blend_weight)
    else
      self.target:add(self.blend_weight, self.G)
    end
  elseif self.mode == 'loss' then
    self.loss = self.strength * self.crit:forward(self.G, self.target)
  end
  self.output = input
  return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
  if self.mode == 'loss' then
    local dG = self.crit:backward(self.G, self.target)
    dG:div(input:nElement())
    self.gradInput = self.gram:backward(input, dG)
    if self.normalize then
      self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
    end
    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
  else
    self.gradInput = gradOutput
  end
  return self.gradInput
end

'ADD TVLoss to the nn'
'total-variation (TV) regularization helps to smooth the image.'
local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(strength)
  parent.__init(self)
  self.strength = strength
  self.x_diff = torch.Tensor()
  self.y_diff = torch.Tensor()
end

function TVLoss:updateOutput(input)
  self.output = input
  return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  local C, H, W = input:size(1), input:size(2), input:size(3)
  self.x_diff:resize(3, H - 1, W - 1)
  self.y_diff:resize(3, H - 1, W - 1)
  self.x_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.x_diff:add(-1, input[{{}, {1, -2}, {2, -1}}])
  self.y_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.y_diff:add(-1, input[{{}, {2, -1}, {1, -2}}])
  self.gradInput[{{}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
  self.gradInput[{{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
  self.gradInput[{{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end



local function main(params)
  '''
  MAIN
  '''
  'Load CNN'
  local cnn = load cnn

  'Load content features image, scale it to 512'
  local content_image = image.load(params.content_image, 3)
  content_image = image.scale(content_image, params.image_size, 'bilinear')

  'Preprocess img: mean pixels, rearrange RGB to BGR'
  local content_image_caffe = preprocess(content_image):float()

  'Style scale. Defaults to 1 x Image Size'
  local style_size = math.ceil(params.style_scale * params.image_size)
  
  'STYLE IMAGE LIST FOR MULT STYLES. RETURNED A DICT OF PROCESSED STYLE IMAGES'
  local style_image_list = params.style_image:split(',')
  local style_images_caffe = {load, scale, preprocess style images and store here}

  'Load init image (content image)'
  '**NOT SURE WHAT DIFF IS OF THIS VS CONTENT IMG LOADED ABOVE**'
  init_image = load, scale, preprocess(init_image):float()
  
  'Dictionary of style blend weights for each style img. Defaults to 1 for each style, then, normalized so all add to 1'
  local style_blend_weights = {} <-- table.insert(style_blend_weights, 1.0)--

  'Activation functions'
  local content_layers = relu4_2 <--layer names to use for content reconstruction. Default is relu4_2.
  local style_layers = relu1_1,relu2_1,relu3_1,relu4_1,relu5_1<--list of layer names to use for style reconstruction

  -- Set up the network, inserting style and content loss modules
  local content_losses, style_losses = {}, {}
  local next_content_idx, next_style_idx = 1, 1
  local net = nn.Sequential()

  'Weight of total-variation (TV) regularization; this helps to smooth the image. Default is 1e-3. Set to 0 to disable TV regularization.'
  tv_mod = nn.TVLoss(params.tv_weight):type(dtype)
  net:add(tv_mod)

  '''
  BUILD CNN
  '''
  for i = 1, #cnn do
    'i, next_content_idx, next_style_idx all start = 1'
    if next_content_idx <= #content_layers or next_style_idx <= #style_layers then
      'get the layer, layer name and layer type'
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)

      'AVE OR MAX POOL LAYERS:'
      'BOOL: is_pooling'
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      'IF PARAM SET, REPLACE ALL MAX POOL WITH AVE POOL'
      net:add(avg_pool_layer) <--adds ave pool layer
      'ELSE, ADD DEFAULT MAX POOL'
      net:add(layer) <--add max pool layer

      'CONTENT LAYERS || IF CURRENT LAYER IS CONV4:'
      if name == content_layers[next_content_idx] then
        print("Setting up content layer", i, ":", layer.name)
        local norm = L1 norm <--param: style and content gradients from each layer will be L1 normalized

        'CONTENT LOSS MODULE:'
        '''
        self.strength = content_weight (default is 5)
        self.target = torch.Tensor()
        self.normalize = L1 norm, or false
        self.loss = 0
        self.crit = MEAN SQUARED ERROR CRITERION
        self.mode = none
        '''
        local loss_module = nn.ContentLoss(params.content_weight, norm):type(dtype)
        'ADD CUSTOM LOSS LAYER TO NET'
        net:add(loss_module)
        '**ADD CUSTOM LOSS TO DICTIONARY CALLED "CONTENT_LOSSES" **'
        table.insert(content_losses, loss_module)
        'go to next layer'
        next_content_idx = next_content_idx + 1
      end --end if content layer

      'STYLE LAYERS || IF CURRENT LAYER IS  CONV 1,2,3,4,5'
      if name == style_layers[next_style_idx] then
        print("Setting up style layer  ", i, ":", layer.name)
        local norm = L1 norm <--param: style and content gradients from each layer will be L1 normalized
        'STYLE LOSS MODULE'
        '''
        self.normalize = L1 or false
        self.strength = style_weight,  Default is 1e2
        self.target = torch.Tensor()
        self.mode = none
        self.loss = 0
        self.gram = nn.GramMatrix()
        self.blend_weight = nil
        self.G = nil
        self.crit = MEAN SQUARED ERROR CRITERION
        '''
        '**NOT SURE ABOUT GramMatrix()**'
        local loss_module = nn.StyleLoss(params.style_weight, norm):type(dtype)
        'ADD CUSTOM LOSS LAYER TO NET'
        net:add(loss_module)
        '**ADD CUSTOM LOSS TO DICTIONARY CALLED "CONTENT_LOSSES" **'
        table.insert(style_losses, loss_module)
        'go to next layer'
        next_style_idx = next_style_idx + 1
      end --end if style layer
    end --end if next_content_idx <= #content_layers or next_style_idx <= #style_layers
  end -- end cnn builder
  '''
  END CNN BUILDER
  '''

  '''SETS "MODE" IN ALL CONTENT LOSS LAYERS TO "CAPTURE" '''
  -- Capture content targets
  print 'Capturing content targets'
  for i = 1, #content_losses do
    content_losses[i].mode = 'capture'
  end
  print(net)

  '''
  FOWARD PASS WITH CONTENT IMAGE
  '''
  '------------------->>>>>>>>>>'
  net:forward(content_image_caffe:type(dtype))
  '------------------->>>>>>>>>>'

  ''' RESET ALL CONTENT LOSS LAYES TO "NONE" '''
  -- Capture style targets
  for i = 1, #content_losses do
    content_losses[i].mode = 'none'
  end
  ''' SETS ALL STYLE LOSS LAYES TO "CAPTURE" '''
  'Note: modified for single style image'
  print(string.format('Capturing style target'))
  for j = 1, #style_losses do
    style_losses[j].mode = 'capture'
    style_losses[j].blend_weight = style_blend_weights
  end


  '''
  FOWARD PASS WITH STYLE IMAGE
  '''
  '------------------->>>>>>>>>>'
  net:forward(style_images_caffe:type(dtype))
  '------------------->>>>>>>>>>'


  ''' SET ALL CONTENT AND STYLE LOSS LAYERS MODE TO "LOSS" '''
  -- Set all loss modules to loss mode
  for i = 1, #content_losses do
    content_losses[i].mode = 'loss'
  end
  for i = 1, #style_losses do
    style_losses[i].mode = 'loss'
  end

  -- We don't need the base CNN anymore, so clean it up to save memory.
  '''CODE TO CLEAR GPU USED IN BASE CNN'''

  -- Initialize the image
  '''EITHER RAND IMAGE OR LOADED CONTENT IMAGE'''
  img = torch.randn(content_image:size()):float():mul(0.001)
  '''OR'''
  img = init_image:clone()
  '''OR'''
  img = content_image_caffe:clone()

  '''RUN LOADED CONTENT IMG THROUGH NETWORK'''
  -- Run it through the network once to get the proper size for the gradient
  '------------------->>>>>>>>>>'
  local y = net:forward(img)
  '------------------->>>>>>>>>>'

  -- All the gradients will come from the extra loss modules, so we just pass
  -- zeros into the top of the net on the backward pass.
  local dy = img.new(#y):zero()

  -- Declaring this here lets us access it in maybe_print
  '''SLIM TO JUST USE ADAM'''
  optim_state = {
    learningRate = params.learning_rate,
  }

  local function maybe_print(t, loss)
  '''PER ITERATION, PRINT CONTENT LOSS, STYLE LOSS, AND TOTAL LOSS'''
  end

  local function maybe_save(t)
    '''PER ITERATION SAVE MODEL OUTPUTS'''
    local disp = deprocess(img:double())
    disp = image.minmax{tensor=disp, min=0, max=1}
    local filename = build_filename(params.output_image, t)
  end

  -- Function to evaluate loss and gradient. We run the net forward and
  -- backward to get the gradient, and sum up losses from the loss modules.
  -- optim.lbfgs internally handles iteration and calls this function many
  -- times, so we manually count the number of iterations to handle printing
  -- and saving intermediate results.
  local num_calls = 0
  local function feval(x)
    num_calls = num_calls + 1
    '------------------->>>>>>>>>>'
    net:forward(x)
    '------------------->>>>>>>>>>'
    '''
    function ContentLoss:updateGradInput(input, gradOutput)
      if self.mode == "loss" then
        if input:nElement() == self.target:nElement() then
          self.gradInput = self.crit:backward(input, self.target)
        end
        self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
        self.gradInput:mul(self.strength)
        self.gradInput:add(gradOutput)
      else <-- mode is not "loss"
        self.gradInput:resizeAs(gradOutput):copy(gradOutput)
      end 
      return self.gradInput
    end
    '''   

    local grad = net:updateGradInput(x, dy) --LOOK INTO THIS
    local loss = 0
    for _, mod in ipairs(content_losses) do
      loss = loss + mod.loss '<-- Adds up loss for all CONTENT loss layers'
    end
    for _, mod in ipairs(style_losses) do
      loss = loss + mod.loss '<-- Adds up loss for all STYLE loss layers'
    end
    maybe_print(num_calls, loss)
    maybe_save(num_calls)

    -- optim.lbfgs expects a vector for gradients
    return loss, grad:view(grad:nElement())
  end

  -- Run optimization.
  '''SLIM DOWN: JUST USE ADAM'''
  for t = 1, params.num_iterations do <-- set to 1000 by default
    local x, losses = optim.adam(
                                feval, --runs fwd pass, add up loss, returns loss and  XXX
                                img, --content image
                                optim_state -- learning rate, default 1e1
                              )
  end
end
'''!! END MAIN !!'''

'''
function ContentLoss:updateOutput(input)
  if self.mode == 'loss' then
    self.loss = self.crit:forward(input, self.target) * self.strength
  elseif self.mode == 'capture' then
    self.target:resizeAs(input):copy(input)
  end
  self.output = input
  return self.output
end
'''