# from pathlib import Path
# from typing import Sequence
import torch
import numpy as np
import torch.nn.functional as F
import copy 
# from albumentations.pytorch import ToTensorV2
# import albumentations as A
# import cv2
# from argparse import Namespace
# import nibabel as nib
import sys
import os 
##################################
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

import sam2 
from sam2.utils.transforms import SAM2Transforms
from typing import List, Optional, Tuple, Union
########################################
from monai.data import MetaTensor 
import re
from itertools import product 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings 
import logging

############################# Adding a path to the checkpoint in order to check whether a checkpoint is available, or for storage of a downloaded checkpoint. 
ckpt_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'ckpt')



#Sanity checking:

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def sanity_check(im_slice:np.ndarray, prompts:dict, transpose=False):
    try: 
        points_npy = torch.cat(prompts['points'], dim=0).numpy()
        points_lbs = torch.cat(prompts['points_labels'], dim=0).numpy()
    except:
        pass 
    try:
        scribbles_npy = torch.cat(prompts['scribbles'], dim=0).numpy()
        scribbles_lbs = torch.cat(prompts['scribbles_labels'], dim=0).numpy() 
    except:
        pass
    try:
        bbox_npy = [i[0].numpy() for i in prompts['bboxes']]
    except:
        pass 
    if transpose:
        im_slice = im_slice.T
    im_slice = ((im_slice - im_slice.min())/(im_slice.max() - im_slice.min() + 1e-6))[...,np.newaxis]
    im_slice = np.repeat(im_slice, 3, axis=-1)
    plt.figure()
    plt.imshow(im_slice)
    try:
        show_points(points_npy, points_lbs, plt.gca())
    except:
        pass
    try:
        show_points(scribbles_npy, scribbles_lbs, plt.gca())
    except:
        pass 
    
    try:
        for box in bbox_npy:
            show_box(box, plt.gca())
    except:
        pass 
    print(f'plotted prompts according to RAS coordinates order on image slice which has transposed property {transpose} inside the plotting function.')
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f'RAS_prompts_on_plotting_img_transpose_{transpose}.png'))
    plt.close()


def sanity_check_output(im_slice:np.ndarray, im_type:str, prompts:dict, transpose=False):
    try: 
        points_npy = torch.cat(prompts['points'], dim=0).numpy()
        points_lbs = torch.cat(prompts['points_labels'], dim=0).numpy()
    except:
        pass
    try:
        scribbles_npy = torch.cat(prompts['scribbles'], dim=0).numpy()
        scribbles_lbs = torch.cat(prompts['scribbles_labels'], dim=0).numpy() 
    except:
        pass
    try:
        bbox_npy = [i[0].numpy() for i in prompts['bboxes']]
    except:
        pass 
    if transpose:
        im_slice = im_slice.T
    im_slice = ((im_slice - im_slice.min())/(im_slice.max() - im_slice.min() + 1e-6))[...,np.newaxis].cpu()
    im_slice = np.repeat(im_slice, 3, axis=-1)
    plt.figure()
    plt.imshow(im_slice)
    try:
        show_points(points_npy, points_lbs, plt.gca())
    except:
        pass 
    try:
        show_points(scribbles_npy, scribbles_lbs, plt.gca())
    except:
        pass    
    try:
        for box in bbox_npy:
            show_box(box, plt.gca())
    except:
        pass 
    print(f'plotted output probability map which has transposed property {transpose} inside the plotting function.')
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f'output_{im_type}_map_transpose_{transpose}.png'))
    plt.close()
    
def sanity_check_post_map(im_slice:torch.Tensor, prompts:dict, transpose=False):
    #We assume a slightly different structure as the image slice and prompts are assumed to be ready for injection into the network. So we need modify func above.
    try:
        points_npy = [i.numpy() for i in torch.unbind(prompts['points'], axis=0)]
        points_lbs = [i.numpy() for i in torch.unbind(prompts['points_labels'], axis=0)]
    except:
        pass 
    try:
        scribbles_npy = [i.numpy() for i in torch.unbind(prompts['scribbles'], axis=0)]
        scribbles_lbs = [i.numpy() for i in torch.unbind(prompts['scribbles_labels'],axis=0)]
    except:
        pass
    try:
        #bbox is not in a list now, it comes in a N_box x 2 x 2 shape torch tensor.
        bbox_npy = [i.numpy().flatten() for i in torch.unbind(prompts['bboxes'], axis=0)]
    except:
        pass 
    
    im_slice_npy = np.moveaxis(im_slice[0].numpy(), 0, -1)
    if transpose:
        im_slice_npy = np.swapaxes(im_slice_npy, 0, 1) #Standard transposition will actually send the rgb channels back into axis=0, so lets not do this! :)
    im_slice_npy = ((im_slice_npy - im_slice_npy.min())/(im_slice_npy.max() - im_slice_npy.min() + 1e-6))
    plt.figure()
    plt.imshow(im_slice_npy)
    
    try:
        for p, p_lb in zip(points_npy, points_lbs):
            show_points(p, p_lb, plt.gca())
    except:
        pass 
    try:
        for p, p_lb in zip(scribbles_npy, scribbles_lbs):
            show_points(p, p_lb, plt.gca())
    except:
        pass
    try:
        for box in bbox_npy:
            show_box(box, plt.gca())
    except:
        pass 
    print(f'plotted prompts according to mapped (resized) RAS coordinates order on resized image slice which has transposed property {transpose} inside the plotting function.')
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f'model_dom_prompts_on_plotting_img_transpose_{transpose}.png'))
    plt.close()

     


huggingface_model_id_to_filenames = {
    "facebook/sam2-hiera-tiny": (
        "sam2_hiera_t.yaml", 
        "sam2_hiera_tiny.pt"
        ),
    "facebook/sam2-hiera-small": (
        "sam2_hiera_s.yaml", 
        "sam2_hiera_small.pt"
        ),
    "facebook/sam2-hiera-base-plus": (
        "sam2_hiera_b+.yaml",
        "sam2_hiera_base_plus.pt",
        ),
    "facebook/sam2-hiera-large": (
        "sam2_hiera_l.yaml",
        "sam2_hiera_large.pt"
        ),
}

def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cpu",#"cuda"
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_hf(model_id, device, **kwargs): #Slight modification to return some of the details that would be desirable for storage in an output log file.

    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = huggingface_model_id_to_filenames[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name, cache_dir=ckpt_dir)
    return config_name, checkpoint_name, ckpt_path, build_sam2(config_file=config_name, ckpt_path=ckpt_path, device=device,**kwargs)


class InferApp:
 
    def __init__(self, dataset_info, infer_device):
        
        self.sanity_check = True
        self.sanity_slice_check = 39
        #Some hardcoded params only for performing a sanity check on mapping the input image domain to the domain expected by the model.

        ############ Initialising the inference application #####################
        self.dataset_info = dataset_info
        self.infer_device = infer_device

        if self.infer_device.type != "cuda":
            raise RuntimeError("SAM2 should only be run on cuda.")

        #Setting image configurations which will be used for configuring the sam model. 
        
        #Setting the names for the corresponding indices of the input image arrays (we will assume that the inputs will be oriented in RAS convention)
        
        index_to_plane = {
            0:'sagittal',
            1:'coronal',
            2:'axial'
        }

        
        image_axes = (2,) 
        
        if any([i!=2 for i in image_axes]):
            warnings.warn('Image axes != 2 selected, but the operations performed to map the coordinates into the model domain are extrapolated from ax=2, major warning.')
        if len(image_axes) > 1:
            raise Exception('There is no existing strategy for performing fusion of segmentations across different planes, only a singular plane can be processed.')
        
        model_id = "facebook/sam2-hiera-large"

        self.app_params = {
            'model_id' : model_id,  
            'image_axes': {k:index_to_plane[k] for k in image_axes}
        } 
        #Loading inference model. Huggingface will determine whether it is necessary to download the model by examining the contents of the checkpoint directory/cache.
        self.load_model()
        
        if self.model.training:
            raise Exception('The model must be in eval mode for inference')
        
        self.app_params.update({'image_size': (self.model.image_size, self.model.image_size)}) 



        self.build_inference_apps()

        # ########################################################################## 

        #Some assumptions we make are listed below when they have an actionable behaviour. Some however, are not: E.g., one assumption we make is that while we will
        #enforce that the bbox will remain static post-introduction of a bbox to a slice (and for which we remove any repeats since it would constitute the generation of extra
        #instances), we do not do this for the points/scribbles. There may be instances where a user is with insistence trying to repeatedly click on the same point!
        #
        # Implicitly we are using the original assumption of the model, which is that bboxes are strictly for constraining, and not for editing!

        #Initialising any remaining variables required for performing inference.

        self.autoseg_infer = True #This is a variable for storing the action taken in the instance where there is no prompting information provided in a slice.
        #In the case where it is True, a prediction will be made, and the stored pred and output pred will be the same.
        #In the case where it is False, a prediction will not be made, the stored pred will be None, and the output pred will be zeroes.
        
        self.static_bbox = True #This is a variable denoting whether we will be permitting for the use of bboxes which are dynamic throughout refinement process. 
        #NOTE: This is within a given slice, it is entirely plausible that the set of bboxes would change throughout if annotation was occuring on a slice-by-slice basis.

        #In the case where it is False, it would indicate that the provided bboxes can be dynamic post slice-level initialisation (at the slice level, clearly for a 
        # volume a slice-by-slice method could be dynamic in the sense that the slice being segmented on could change!) between iterations.
        #In the case where it is True, it must be static (at the slice level) between iterations. I.e., if it is not the first set of bboxes in that slice it should 
        # raise an exception. 

        # self.split_forward_mask = True #This is a variable denoting whether the lowres mask that is forward propagated (in cases where we are actually doing this) will
        # #be split by bbox quantity. 

            #Deprecated, SAM2 enables batched inference and so enables instance-wise generation in a straightforward manner using their demo code.
        

        self.prop_freeform_uniformly = True #This is a variable denoting whether the free-form prompts, like clicks and scribbles are uniformly distributed across
        #the set of instance-level closed contour prompts (i.e. bbox) under the assumption that we are working under semantic segmentation constraints rather than instance segmentation.
        # I.e., that a single class could have multiple instances, and as such the free-form prompts are not as strictly separated as a bbox. Currently raises an
        # exception if this is false.

        self.multimask_output_always = True  # By default, it will be true to match defaults from demo, this means that masks are always generated by propagating 
        #through each of the MLPs (3) and then chosen according to the best predicted iou. If false, then whether or not to perform this action must be 
        # determined heuristically. Note that this only pertains to cases where prompts ARE provided. If they are not then this will always be on in order to try and 
        #segment (although extremely unlikely to succeed).
        self.permitted_prompts = ('points', 'bboxes', 'scribbles')
        

        ###########################################################################################
        #Setting some parameters for the preprocessing, and postprocessing of the image and predictions. 


        #Self.model._image_size = 1024 (1024 x 1024).
        self.mask_threshold_logits = 0 #Demo value: This mask threshold is for the binarisation of a logits map, ergo...
        self.mask_threshold_sigmoid = 1/(float(1 + np.exp(-self.mask_threshold_logits))) #The sigmoid value is derived from the value of the logits threshold.
        self.max_hole_area = 0 #Demo value: This parameter determines the maximum size of a hole in a connected component which can be closed
        self.max_sprinkle_area = 0# Demo value:This parameter determines the maximum size below which a sprinkle (i.e. a connected component) can be zeroed out.
        
        # Spatial dims for backbone feature maps from sam2 repo/demo.
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        
        #Setting the image preprocessing transformation stack according to the demo.

        self.sam2_transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=self.mask_threshold_logits,
            max_hole_area=self.max_hole_area,
            max_sprinkle_area=self.max_sprinkle_area
        )

    def app_configs(self):
        #STRONGLY Recommended: A method which returns any configuration specific information for printing to the logfile. Expects a dictionary format.
        return self.app_params 
    
    def load_model(self):

        config_name, checkpoint_name, ckpt_path, model = build_sam2_hf(self.app_params['model_id'], device=self.infer_device)

        model_info = {
            'model_config':config_name,
            'ckpt_name': checkpoint_name,
            'ckpt_path': ckpt_path,
        }
        self.app_params.update(model_info)
        self.model = model

    def build_inference_apps(self):
        #Building the inference app, needs to have an end to end system in place for each "model" type which can be passed by the request: 
        # 
        # IS_autoseg, IS_interactive_init, IS_interactive_edit. (all are intuitive wrt what they represent.) 
        
        self.infer_apps = {
            'IS_autoseg':{'binary_predict':self.binary_inference},
            'IS_interactive_init': {'binary_predict':self.binary_inference},
            'IS_interactive_edit': {'binary_predict':self.binary_inference}
            }
    
    
    def binary_inference(self, request):

        warnings.warn('Current binary inference strategy is incapable of disentangling the open contour/free-form prompts for different connected components, unlike a closed contour strategy, as such they are propagated for inference for all components.')
        #Mapping the input request to the model domain:
        init_bool, infer_slices, affine = self.binary_subject_prep(request=request)
        #The input information has been stored separately since we will iteratively update this throughout the refinement process.

        #Extracting prediction slices:
        self.binary_predict(init_bool, infer_slices)

        #Converting the set of prediction and probability map slices into the output volume:
        discrete_mask, prob_mask = self.binary_merge_slices()
        
        return discrete_mask, prob_mask, affine 
    
    def binary_subject_prep(self, request:dict):
        
        #Here we perform some actions for determining the state of the infer call for adjusting some of our info extraction mechanisms.
         
        #Ordering the set of interaction states provided, first check if there is an initialisation: if so, place that first. 
        im_order = [] 
        init_modes  = {'Automatic Init', 'Interactive Init'}
        edit_names_list = list(set(request['im']).difference(init_modes))

        #Sorting this list.
        edit_names_list.sort(key=lambda test_str : list(map(int, re.findall(r'\d+', test_str))))

        #Extending the ordered list. 
        
        im_order.extend(edit_names_list) 
        #Loading the image and prompts in the input-im domain & the zoom-out domain.
        
        if request['model'] == 'IS_interactive_edit':
            key = edit_names_list[-1]
            is_state = request['im'][key]
            if all([i is None for i in is_state['interaction_torch_format']['interactions'].values()]) or all([i is None for i in is_state['interaction_torch_format']['interactions_labels'].values()]):
                raise Exception('Cannot be an interactive request without interactive inputs.')
            init = False 
            
            assert isinstance(self.image_features_dict, dict) and self.image_features_dict
            assert isinstance(self.internal_lowres_mask_storage, dict) and self.internal_lowres_mask_storage
            assert isinstance(self.internal_discrete_output_mask_storage, dict) and self.internal_discrete_output_mask_storage
            assert isinstance(self.internal_prob_output_mask_storage, dict) and self.internal_prob_output_mask_storage
            assert isinstance(self.orig_prompts_storage_dict, dict) and self.orig_prompts_storage_dict
            assert isinstance(self.model_prompts_storage_dict, dict) and self.model_prompts_storage_dict #Just asserting that these are dicts and also non-empty.
            assert isinstance(self.box_prompted_slices, dict) and self.box_prompted_slices 

        elif request['model'] == 'IS_interactive_init':
            key = 'Interactive Init' 
            is_state = request['im'][key]
            if all([i is None for i in is_state['interaction_torch_format']['interactions'].values()]) or all([i is None for i in is_state['interaction_torch_format']['interactions_labels'].values()]):
                raise Exception('Cannot be an interactive request without interactive inputs.')
            init = True 

            self.image_features_dict = dict()
            self.internal_lowres_mask_storage = dict() #This is in the model domain!
            self.internal_discrete_output_mask_storage = dict() #This is in the cv2 coordinate system, 
            #but otherwise in the domain at the slice level of the image (i.e., same dimensions for the corresponding coordinate axes, just flipped.)
            self.internal_prob_output_mask_storage = dict() #This is in the cv2 coordinate system, 
            #but otherwise in the domain at the slice level of the image (i.e., same dimensions for the corresponding coordinate axes, just flipped.)
            
            self.box_prompted_slices = dict() #Stores the set of slices which have been prompted already, this is relevant for the bbox placement as it will be assumed
            #that in cases where it is static, that incoming bbox prompts will be cross-examined accordingly.
            self.orig_prompts_storage_dict = dict()
            self.model_prompts_storage_dict = dict()

        elif request['model'] == 'IS_autoseg':
            key = 'Automatic Init'
            is_state = request['im'][key]
            if is_state is not None:
                raise Exception('Autoseg should not have any interaction info.')
            init = True 

            self.image_features_dict = dict()
            self.internal_lowres_mask_storage = dict() #This is in the model domain!
            self.internal_discrete_output_mask_storage = dict() #This is in the cv2 coordinate system, 
            #but otherwise in the domain at the slice level of the image (i.e., same dimensions for the corresponding coordinate axes, just flipped.)
            self.internal_prob_output_mask_storage = dict() #This is in the cv2 coordinate system, 
            #but otherwise in the domain at the slice level of the image (i.e., same dimensions for the corresponding coordinate axes, just flipped.)
            
            self.box_prompted_slices = dict() #Stores the set of slices which have been box-prompted already, this is relevant for the bbox placement as it will be assumed
            #that in cases where it is static, that incoming bbox prompts will be cross-examined accordingly.
            self.orig_prompts_storage_dict = dict()
            self.model_prompts_storage_dict = dict() 
        #Mapping image and prompts to the model's coordinate space. NOTE: In order to disentangle the validation framework from inference apps 
        # this is always assumed to be handled within the inference app.

        infer_slices = self.binary_prop_to_model(request['image'], is_state, init)        
        #The images, image embeddings, and prompt information are all stored in attributes separately. 
        for ax in self.app_params['image_axes']:
            print(f'altered slices for ax {ax} are {infer_slices[ax]}')

        affine = request['image']['meta_dict']['affine']
        return init, infer_slices, affine  

    
    def binary_prop_to_model(self, im_dict: dict, is_state: dict | None, init: bool):
        
        #Prompts and images are provided in R ->L, A->P, S -> I  (where the image itself was also been correspondingly rotated since array_coords >=0).
        # 
        #Visual inspection of the images provided in the demo demonstrate that the positive directions of the axes in the axial slice corresponds to the R -> L, A -> P convention.
        #but, the ordering of the axes differs. I.e., the y dimension of the image array is the A -> P dimension, while the x dimension is the R -> L dimension.

        #Note that OpenCV convention is -----> x, therefore an array that is M x N represents N in the X direction, and M in the y direction.
                                    #  |
                                    #  |
                                    #  |
                                    #  v
                                    # y

        #We first propagate the image into the model domain, and store the image embeddings.

        if init:
            input_dom_img = im_dict['metatensor']
            # input_dom_affine = im_dict['meta_dict']['affine']
            # input_dom_shape = input_dom_img.shape[1:] #Assuming a channel-first image is being provided.
            im_slices_model_dom, input_dom_shapes = self.binary_im_to_model_dom(input_dom_img) 
            #Storing the shape of the slice in each corresponding axis being used.
            self.orig_im_shape = input_dom_img.shape[1:]
            self.input_dom_shapes = input_dom_shapes
            #Storing the image slice in the input model domain in memory for sanity checks 
            if self.sanity_check:
                self.im_slices_post_map = im_slices_model_dom 
            #Now we will extracted the image features correspondingly, and store them in memory.
            self.binary_extract_im_features(im_slices_model_dom=im_slices_model_dom)

        #Now we propagate the prompt information into the model domain. 

        if bool(is_state):
            p_dict = (is_state['interaction_torch_format']['interactions'], is_state['interaction_torch_format']['interactions_labels'])
            
            if p_dict[0]['bboxes'] is not None and p_dict[1]['bboxes_labels'] is not None:
                #We will remove any background bboxes here. SAM2 cannot handle these (nor does it have any meaning in the context within which they use this.)
                if not any([i == 1 for i in p_dict[1]['bboxes_labels']]): 
                    p_dict[0]['bboxes'] = None
                    p_dict[1]['bboxes_labels'] = None 
                else:
                    #There is at least one non-background bbox. See if there are any background bboxes, and find their indices. Then delete accordingly.
                    bbox_list = []
                    bbox_lb_list = []
                    for i,j in zip(p_dict[0]['bboxes'], p_dict[1]['bboxes_labels']):
                        if j == 1:
                            bbox_list += [i] 
                            bbox_lb_list += [j] 
                    p_dict[0]['bboxes'] = bbox_list
                    p_dict[1]['bboxes_labels'] = bbox_lb_list
                    assert p_dict[0]['bboxes'] != [] and p_dict[1]['bboxes_labels'] != []
            #Determine the prompt type from the input prompt dictionaries: Not sure if intersection is optimal for catching exceptions here.
            provided_ptypes = list(set([k for k,v in p_dict[0].items() if v is not None]) & set([k[:-7] for k,v in p_dict[1].items() if v is not None]))
            
            #
            if any([p not in self.permitted_prompts for p in provided_ptypes]):
                raise Exception(f'Non-permitted prompt was supplied, only the following prompts are permitted {self.permitted_prompts}')
        else:
            #Handling empty prompt dict and/or Autosegmentation.
            provided_ptypes = None
            p_dict = None 
            # self.binary_extract_prompts(current_prompts=None, init_bool=init)
        
        infer_slices = self.binary_extract_prompts(current_prompts=p_dict, provided_ptypes=provided_ptypes, init_bool=init)

        if self.sanity_check:
            for ax in self.app_params['image_axes']:
                sanity_check(self.im_slices_input_dom[ax][self.sanity_slice_check], self.orig_prompts_storage_dict[ax][self.sanity_slice_check], transpose=False)
                sanity_check_post_map(self.im_slices_post_map[ax][self.sanity_slice_check], self.model_prompts_storage_dict[ax][self.sanity_slice_check], transpose=False)

        return infer_slices

    def binary_im_to_model_dom(self, input_dom_im): #Similar logic to that borrowed from RadioActive SAMMed-2D inferer, some modifications were made to accomodate for SAM2's
        #transformation stack. 

        #Assuming that input image is in RAS convention, the axes denote the subset from (0,1,2) which denotes the axes along which slices will be taken, i.e., 
        #if (2), then the values being extracted are from the first 2 according to the third index (i.e. in the R-L/A-P plane, aka axial slices)
        
        #First removing the channel dimension 
        input_dom_im_backend = copy.deepcopy(input_dom_im)[0,:].numpy()

        if self.sanity_check:
            self.im_slices_input_dom = {} #This is a variable that is not used for anything other than sanity_checking that the array channels are in good order. 
        slices_processed = {}
        orig_im_dims = {}

        if len(self.app_params['image_axes']) > 1:
            raise Exception('Implementation currently is not capable of simultaneous handling of > 1 planar segmentations.')
        for ax in self.app_params['image_axes']:
            if self.sanity_check:
                ax_slices_pre_resizing = {}
            ax_slices_process = {} 

            #This normalisation logic is borrowed from RadioActive, as SAM2 does not provide their own preprocessing scripts aside from what is assumed for 
            # SAM (we assume this is done externally). The logic is fairly standard, and not specialised for specific datasets, nor is it most careful about ensuring the 
            # foreground is minimally shifted in intensity.
            for slice_idx in range(input_dom_im_backend.shape[ax]):
                if ax == 0:
                    slice = input_dom_im_backend[slice_idx, :, :]
                elif ax == 1:
                    slice = input_dom_im_backend[:, slice_idx, :]
                elif ax == 2:
                    slice = input_dom_im_backend[:, :, slice_idx]
                else:
                    raise Exception('Cannot have more than three spatial dimensions for indexing the slices, we only permit 3D volumes at most!')
                try:
                    lower_bound, upper_bound = np.percentile(slice[slice > 0], 0.5), np.percentile(slice[slice > 0], 99.5) 
                except:
                    lower_bound, upper_bound = 0, 0

                #We transpose the slice spatially, since RAS orientation does not align with the image array orientation of the demo, we assume self-consistency, this assumption may
                #not be valid but it is our only presumption given the provided information.
                slice = slice.T

                if self.sanity_check:
                    ax_slices_pre_resizing[slice_idx] = slice #We save this to help with our sanity checks.

                #Clamping the voxel intensities.
                slice = np.clip(slice, lower_bound, upper_bound)
                slice = np.round((slice - slice.min()) / (slice.max() - slice.min() + 1e-6) * 255).astype(
                    np.uint8
                )  # Mapping to [0,255] rgb scale
                slice = np.repeat(slice[..., None], repeats=3, axis=-1) #RGB
                
                slice = self.sam2_transforms(slice) #Applying the SAM2 Transforms stack 
                slice = slice[None, ...] #Adding the batch dim such that inference can be performed.

                ax_slices_process[slice_idx] = slice #Adding the slice torch tensor to the dictionary which will be fed forward.

            #Insertion into the dictionary of slices which contains sets of slices/affine planes orthogonal to the axis along which it was marching. 
            slices_processed[ax] = ax_slices_process #
            
            if self.sanity_check:
                self.im_slices_input_dom[ax] = ax_slices_pre_resizing 

            #saving of the original dimensions for the slice extracted, will be required for mapping back to the segmentation space.. #NOTE: Since we have transposed,
            #mapping the predictions back will first require transposing this shape, performing the inverse map, then transposition to return back to RAS space.
            orig_im_dims[ax] = np.array([input_dom_im_backend.shape[i] for i in set(list(range(input_dom_im_backend.ndim))) ^ set([ax])])
            assert orig_im_dims[ax].size == 2 and orig_im_dims[ax].ndim == 1
        return slices_processed, orig_im_dims


    @torch.no_grad()
    def binary_extract_im_features(self, im_slices_model_dom:dict):
        #Function which is designed to extract the image embeddings of the set of input slices in the model domain, stores them for the entire duration of the iterative refinement
        #process. Borrows heavily from the SAM2 predictor class method: set_image.
        
        logging.info("Computing image embeddings for the provided image...")
        for ax in self.app_params['image_axes']:
            axis_embeddings = dict() 
            for slice_idx, img_slice in im_slices_model_dom[ax].items():
            # with torch.no_grad():
                img_slice = img_slice.to(device=self.infer_device)

                assert (
                    len(img_slice.shape) == 4 and img_slice.shape[1] == 3
                ), f"input_image must be of size 1x3xHxW, got {img_slice.shape}"
                
                backbone_out = self.model.forward_image(img_slice)
                _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
                # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
                if self.model.directly_add_no_mem_embed:
                    vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

                feats = [
                    feat.permute(1, 2, 0).view(1, -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
                ][::-1]
                
                axis_embeddings[slice_idx] = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

            self.image_features_dict[ax] = axis_embeddings 

        logging.info("Image embeddings computed.")




#######################################################

#Prompt extraction:

    def binary_extract_prompts(self, current_prompts:dict | None, provided_ptypes:list | None, init_bool:bool):
        #Takes the current prompts, which will need to be added to the set of stored prompts.
        #Prompts and images are provided in R ->L, A->P, S -> I  (where the image itself was also been correspondingly rotated since array_coords >=0).
        

        #Visual inspection of the images provided in the demo demonstrate that the positive directions of the axes in the axial slice corresponds to the R -> L, A -> P convention.
        #but, the ordering of the axes differs. I.e., the y dimension of the image array is the A -> P dimension, while the x dimension is the R -> L dimension.

        #Note that OpenCV convention is -----> x, therefore an array that is M x N represents N in the X direction, and M in the y direction, i.e. y, x ordering.
                                    #  |
                                    #  |
                                    #  |
                                    #  v
                                    # y

        #This therefore requires transposition of the axes for the images when mapping to the model domain. We assume this is consistent
        #across the axes chosen for slicing, but we will only test with the axial slices just to be careful.

        #For the prompt coordinates, they are provided x,y ordering. This corresponds to the original RAS order, so we do nothing.

        #NOTE: However, since we will have to perform a rescaling for insertion to the SAM2 model domain, we will need to perform a rescaling with the scale factors
        #being computed on the transposed image slice coordinates.

        #First we stash the set of prompts in our collected bank in the input image domain.
        infer_slices = self.binary_store_prompts(current_prompts=current_prompts, provided_ptypes=provided_ptypes, init_bool=init_bool)
        
        if init_bool:
            for ax in self.app_params['image_axes']:        
                if self.orig_im_shape[ax] != len(infer_slices[ax]):
                    raise Exception(f'The quantity of altered slices in the initialisation for axis {ax} was {len(infer_slices[ax])}, but it needs to be {self.orig_im_shape[ax]}')
        
        #Now we need to map this into the domain of the expected images, which are defined by the array definition of cv2, and then map that to the model domain.
        self.binary_map_prompt_to_model(infer_slices=infer_slices, init_bool=init_bool)
        return infer_slices
    
    def binary_store_prompts(self, current_prompts: dict | None, provided_ptypes:list | None, init_bool:bool):
        #This function stores the prompts in the original image domain, but according to the corresponding slicing axes.
        if init_bool:
            infer_slices = dict()

            for ax in self.app_params['image_axes']:
                infer_slices[ax] = list(range(len(self.image_features_dict[ax]))) #All slices are going to be used for inference for initialisation!
                self.box_prompted_slices[ax] = {k:False for k in range(len(self.image_features_dict[ax]))} #We start with the assumption that a slice is not bbox prompted,
                #and update this corresponding to the observation. This is required for cases where the assumed structure of a bbox interaction is that of a "grounding"
                #spatial prior.
                
                #Init the dict for the given axis always
                current_ax_dict = dict()
                for slice_idx in range(len(self.image_features_dict[ax])):
                    #In this case we are going to initialise an empty set of prompts for each axis and slice, across all prompt types. 
                    current_ax_dict[slice_idx] = {k:[] for k in list(self.permitted_prompts) + [f'{i}_labels' for i in self.permitted_prompts]}
                self.orig_prompts_storage_dict[ax] = current_ax_dict
                #Initialising the storage dict for the prompts!
                
        if current_prompts is not None: #We could have used provided ptypes list = None also, this denotes the autoseg case!
            if not init_bool:
                infer_slices = {ax:[] for ax in self.app_params['image_axes']} 
                #We create a dict to store the set of slices which were modified for editing so that we are not performing inference on slices which were not prompted.
            
            for ax in self.app_params['image_axes']:  
                #If prompts are not none, we require special treatment according to the prompt type as bbox might be in 3D and needs to be collapsed to a set of 2D!
                
                box_prompted_slices = [] 
                #Initialisation of an empty list containing the list of slice indices for the given axis which contain new initialisation bbox, this is required 
                # such that after we pass through the set of inputted bbox prompts, we can update our memory bank of the slices which have been bbox initialised. 
                # This is also critical for checking that no new bboxes are being placed if we configure the bboxes as being slice-level static!
                    
                for ptype in provided_ptypes:
                    
                    # print(ptype)
                    if ptype == 'points' or ptype == 'scribbles':
                        #We treat scribbles in a very similar capacity as points, but for now we will keep these as separate items in memory.

                        if current_prompts[0][ptype] is None or current_prompts[1][f'{ptype}_labels'] is None:
                            raise Exception('Uncaught instance of no prompts being available for extraction, should have been flagged or handled earlier!')
                        #Extract the list of prompt items, concatenate together for vectorising indexing.
                        p_cat = torch.cat(current_prompts[0][ptype], dim=0)
                        if ptype == 'points':
                            p_lab_cat = torch.cat(current_prompts[1][f'{ptype}_labels'])
                        else:
                            #In the downstream label extraction code, since we will convert a scribble to a set of points, we must also convert the corresponding labels
                            #too.
                            #First extract the length of each scribble in terms of #N_points, then use repeat interleave to expand the set of labels for a 1-to-1 map of points and labels
                            p_lab_cat = torch.repeat_interleave(torch.tensor(current_prompts[1][f'{ptype}_labels']), torch.tensor([p.shape[0] for p in current_prompts[0][ptype]]))
                        
                        #Denoting the required axes when extracting the prompt locations within a given slice.
                        required_ax = list(set([0,1,2]) - set([ax])) 
                        #Required coord component corresponds to the set difference between [0,1,2] and [ax] (where ax is the axial dimension for which we are extracting slices.)
                        
                        #Extracting the set of valid axis slice coordinates to reduce our search:
                        valid_slices = p_cat[:, ax].unique() 
                        for slice_idx in valid_slices.tolist():
                            #Extract the set of coords which have axis dimension equivalent to the slice index.
                            valid_p_idxs = torch.argwhere(p_cat[:,ax] == slice_idx)
                            
                            if valid_p_idxs.numel():
                                #Setting the slices which require inference.
                                if slice_idx in infer_slices[ax]:
                                    #The logic here, is just to append the slice if the slice was not already being tracked, according to whether a prompt exists within
                                    #the slice. It will NOT skip the insertion of the prompt! Hence why we have used a pass explicitly, rather than a continue!
                                    pass
                                else:
                                    infer_slices[ax] += [slice_idx]

                                if valid_p_idxs.numel() > 1:
                                    ps = [p.unsqueeze(0) for p in p_cat[tuple(valid_p_idxs.T)][:, required_ax].unbind(dim=0)]
                                    self.orig_prompts_storage_dict[ax][slice_idx][ptype] += ps
                                    self.orig_prompts_storage_dict[ax][slice_idx][f'{ptype}_labels'] += [p.unsqueeze(0) for p in p_lab_cat[tuple(valid_p_idxs.T)].unbind(dim=0)]
                                    #These lines are extracting the valid set of point coordinates (or labels) according to the condition, extracting the relevant 
                                    # coordinates or labels for the slice and then unrolling into a list of tensors with shape [1,2] or shape [1] respectively.

                                    if not all([(i[0].numpy() <= self.input_dom_shapes[ax]).all() for i in ps]):
                                        Exception('A prompt fell outside of the bounds of the slice of the image, please check whether the prompt is valid, or whether it has been properly processed.')
                                elif valid_p_idxs.numel() == 1:
                                    ps = [p_cat[tuple(valid_p_idxs.T)][:, required_ax]]
                                    self.orig_prompts_storage_dict[ax][slice_idx][ptype] += ps
                                    self.orig_prompts_storage_dict[ax][slice_idx][f'{ptype}_labels'] += [p_lab_cat[tuple(valid_p_idxs.T)]]
                                    #If using scribbles, it can be the exact same, as a scribble represented as a single point is equivalent to treatment as such!
                                    #, hence it will only correspond to a singular label.
                                    if not all([(i[0].numpy() <= self.input_dom_shapes[ax]).all() for i in ps]):
                                        Exception('A prompt fell outside of the bounds of the slice of the image, please check whether the prompt is valid, or whether it has been properly processed.')
                    elif ptype == 'bboxes':
                        #Denoting the required axes for extracting the prompt locations within a given slice.
                        required_ax = list(set([0,1,2,3,4,5]) - set([ax, ax + 3])) 
                        #Required coord component corresponds to the set difference between [0,1,2,3,4,5] and [ax, ax+3] 
                        # (where ax is the axial dimension for which we are extracting slices, this is because the bbox is provided in min_r, min_a, min_s, max_r,max_a,max_s convention)
                        
                        #Slightly different treatment to the points and scribbles, we loop over the set of bboxes instead as we cannot disentangle the structure of a bbox
                        # for inference like we did for the scribbles..... 
                        for bbox, bbox_label in zip(current_prompts[0][ptype], current_prompts[1][f'{ptype}_labels']):
                            box_extrema = bbox[0]
                            #Catching any cases where a 2D bbox is passed through in a 3D format.
                            extrema_matching = [box_extrema[i] == box_extrema[i+3] for i in range(3)]
                            #If the extrema are matching in the axis we are currently using, great! Just take the bbox of that slice. Otherwise, we continue because
                            #a 2D bounding box is just a line when being sliced along axes orthogonal to the plane of the bbox. 
                            if sum(extrema_matching) > 1:
                                #In this scenario, somehow we have been provided with a 3D bbox that is actually just a line... 
                                # #NOTE: the choice could be made to convert this to a scribble or point but we will not.
                                warnings.warn('Somehow have been provided with a bounding box set of extrema which describes a line or point, please check.')
                                continue
                            elif sum(extrema_matching) == 1:
                                if list(filter(lambda i: extrema_matching[i], range(len(extrema_matching))))[0] != ax:
                                    #In this case, the axis where the extrema were matching was NOT the same axis we are currently inspecting..., so continue,
                                    # otherwise we are just getting a set of lines..... 
                                    # NOTE: the choice could be made to convert this to a scribble but we will not. 
                                    continue 
                                    #We use continue here to skip over this bbox completely! Not a pass.
                                else:
                                    #In this case, the axis was matching, so we can potentially insert at that slice the corresponding bbox according to extrema on the 
                                    # other axes

                                    #We set the slice_idx for this bbox to be given by the value of the "extrema" for the given axis.
                                    slice_idx = int(box_extrema[ax]) 
                                    #We extract the bbox parameterisation.
                                    bbox = box_extrema[[i for i in range(box_extrema.shape[0]) if i not in [ax, ax + 3]]]
                                    #Now we check for any faults and add in any bboxes appropriately. 
                                    if self.static_bbox:
                                        #First we perform a check to see if the corresponding slice has had a set of bboxes it was previously prompted with already, 
                                        # if not then append it to the list of slices. NOTE: Since it is possible for multiple bboxes to be used, the update occurs 
                                        # after all bboxes are processed.
                                        if self.box_prompted_slices[ax][slice_idx]:
                                            #In this case we must perform a check to make sure that the bbox being provided is not different from that which already
                                            #was placed. If it is not distinct then just continue.
                                            if any([torch.all(bbox.unsqueeze(0) == prior_box) for prior_box in self.orig_prompts_storage_dict[ax][slice_idx]['bboxes']]):
                                                continue #If the bbox is already in the stashed set of bboxes, then do not append it. 
                                                # SAMMed2D is incompatible with this and will just keep growing the quantity of prediction masks.
                                                #It is unclear what the procedure should be with a changing quantity of bboxes, as forward pass creates a masks for each bbox. 
                                            else:
                                                #If the bbox was not, and the slice has already been box-prompted, since we are in the static_bbox config we raise an
                                                #exception.
                                                raise Exception(f'A new bounding box was provided in axis {ax}, slice {slice_idx} in an iteration after the initial instance where the slice had been bbox-prompted.')
                                        else:
                                            #In this case then it is perfectly acceptable to just append the bbox prompts because the given slice has not been previously bbox initialised!
                                            #We still check if the bbox hasn't already been placed somehow anyways to prevent repeats!
                                            if any([torch.all(bbox.unsqueeze(0) == prior_box) for prior_box in self.orig_prompts_storage_dict[ax][slice_idx]['bboxes']]):
                                                continue #If the bbox is already in the stashed set of bboxes, then do not append it. 
                                               
                                                #It is unclear what the procedure should be with a changing quantity of bboxes, as forward pass creates a masks for each bbox. 
                                            else:
                                                self.orig_prompts_storage_dict[ax][slice_idx]['bboxes'] += [bbox.unsqueeze(0)] #We will use a 1 x N_dim convention
                                                self.orig_prompts_storage_dict[ax][slice_idx]['bboxes_labels'] += [bbox_label]

                                            #Appending the slice to the set of slices which are bbox initialised, we will later consider only the unique indices so any
                                            #redundancy is not important. We do not update the bool here as we do not want to throw an exception as we loop through bboxes.
                                            box_prompted_slices += [slice_idx] 
                                    else:
                                        raise NotImplementedError('The handling of non-static bboxes (post-initialisation) is not supported within a given slice.')       
                                
                                    if slice_idx in infer_slices[ax]:
                                        #The logic here, is just to append the slice if the slice was not already being tracked, according to whether a new prompt 
                                        # exists within the slice. It will NOT skip the insertion of the prompt! Hence why we have used a pass explicitly, 
                                        # rather than a continue!
                                        pass 
                                    else:
                                        infer_slices[ax] += [slice_idx]
                                    if not all([bbox[i] <= bound and bbox[i + 2] <= bound for i, bound in enumerate(self.input_dom_shapes[ax])]):
                                        raise Exception('A prompt fell outside of the bounds of the slice of the image, please check whether the prompt is valid, or whether it has been properly processed.')
                            else:
                                #In this case, no extrema were matching, we can proceed as standard:

                                for slice_idx in range(box_extrema[ax],box_extrema[ax + 3] + 1):    
                                #Since the bounding box is consistent in shape across the slices along a given axis, we can just insert as expected across all corresponding slices bounded
                                #by the extrema of the bounding box along the given axis.
                                    bbox = box_extrema[[i for i in range(box_extrema.shape[0]) if i not in [ax, ax + 3]]]
                                    
                                    #Now we check for any faults, or add any bboxes by unroll a 3d bbox into 2d slices and checking each slice to see if it has been
                                    #bbox initialised, and whether it has any dynamically generated bboxes within the slice.
                                    if self.static_bbox:
                                        if self.box_prompted_slices[ax][slice_idx]:
                                            if any([torch.all(bbox.unsqueeze(0) == prior_box) for prior_box in self.orig_prompts_storage_dict[ax][slice_idx]['bboxes']]):
                                                continue #If the bbox is already in the stashed set of bboxes, then do not append it. SAMMed2D is incompatible with this,
                                                #and it is unclear what the procedure should be with variable quantities of bboxes as forward pass creates N masks for each bbox. 
                                            else:
                                                raise Exception(f'A new bounding box was provided in axis {ax}, slice {slice_idx} in an iteration after the initial instance where the slice had been bbox-prompted.')
                                        else:
                                            #In this case then it is perfectly acceptable to just append the bbox prompts because the given slice has not been 
                                            # previously bbox initialised! We still check if the bbox hasn't already been placed somehow anyways to prevent repeats!
                                            if any([torch.all(bbox.unsqueeze(0) == prior_box) for prior_box in self.orig_prompts_storage_dict[ax][slice_idx]['bboxes']]):
                                                continue #If the bbox is already in the stashed set of bboxes, then do not append it. 
                                                
                                                #It is unclear what the procedure should be with a changing quantity of bboxes, as forward pass creates a masks for each bbox.
                                            else:
                                                #If it was not bbox initialised in a prior iteration of inference (or if there wasn't one already) then we can freely add
                                                self.orig_prompts_storage_dict[ax][slice_idx]['bboxes'] += [bbox.unsqueeze(0)] 
                                                self.orig_prompts_storage_dict[ax][slice_idx]['bboxes_labels'] += [bbox_label]

                                            #Appending the slice to the set of slices which are bbox initialised, we will later consider only the unique indices so any
                                            #redundancy is not important. We do not update the bool here as we do not want to throw an exception as we loop through bboxes.
                                            box_prompted_slices += [slice_idx] 
                                    else:
                                        raise NotImplementedError('The handling of non-static bboxes (post-initialisation) is not supported within a given slice.') 
                                    if slice_idx in infer_slices[ax]:
                                        #The logic here, is just to append the slice if the slice was not already being tracked, according to whether a prompt exists within
                                        #the slice. It will NOT skip the insertion of the prompt! Hence why we have used a pass explicitly, rather than a continue!
                                        pass 
                                    else:
                                        infer_slices[ax] += [slice_idx]
                                    
                                    if not all([bbox[i] <= bound and bbox[i + 2] <= bound for i, bound in enumerate(self.input_dom_shapes[ax])]):
                                        raise Exception('A prompt fell outside of the bounds of the slice of the image, please check whether the prompt is valid, or whether it has been properly processed.')
                    else:
                        raise Exception('Somehow, a non-supported prompt type managed to get through.')
        
                #Here we commit the stashed set of slices which have been bbox initialised.
                for idx in set(box_prompted_slices): self.box_prompted_slices[ax][idx] = True  
        
        
        for i in self.app_params['image_axes']: infer_slices[i].sort() 
        return infer_slices
    
    def binary_map_prompt_to_model(self, infer_slices: dict, init_bool: bool): #Borrowing the logic from prep_prompts function in the sam2 image_predictor class.
        #This function maps the coordinates in the image domain, to the coordinates in the model domain.

        #The second value in the shape of the img array in the cv2 domain corresponds to the first value in the RAS ordering (e.g. in RA plane, R,   in AS plane, A, 
        # in RS plane, R). The first value in the shape corresponds to the second value in the RAS ordering (e.g. in RA plane, A,  in AS plane, S   in RS plane, R)
        
        #This is reflected in the methods used for performing these transformations called.

        for ax in self.app_params['image_axes']:
            #For each axis, we store the set of prompts in the altered slices.
            if init_bool:
                if self.orig_im_shape[ax] != len(infer_slices[ax]):
                    raise Exception(f'The quantity of altered slices in the initialisation for axis {ax} was {len(infer_slices[ax])}, but it needs to be {self.orig_im_shape[ax]}')
                self.model_prompts_storage_dict = copy.deepcopy(self.orig_prompts_storage_dict)
                #We copy as this will copy over the structure, but this does not mean that our job is yet complete, for autoseg it will though.
            for slice_idx in infer_slices[ax]:
                #This isn't efficient, but we are just going to recompute the mapped values every single time for the set of altered slices 
                # otherwise it would require that we needed to have outline which were the new prompts. 
                if all([i == [] for i in self.orig_prompts_storage_dict[ax][slice_idx].values()]):
                    pass #In this case, just pass over, we copied over the empty list already for the autoseg modes. 
                else:
                    #In this case, there are some prompts which require mapping.
                    for ptype in self.permitted_prompts:
                        if self.orig_prompts_storage_dict[ax][slice_idx][ptype] == []:
                            continue #In this case the given prompt did not contain anything.
                        if ptype == 'points' or ptype == 'scribbles':
                            p, p_lab = self.orig_prompts_storage_dict[ax][slice_idx][ptype], self.orig_prompts_storage_dict[ax][slice_idx][f'{ptype}_labels']
                            if len(p) > 1:
                                p_inp = torch.cat(p, dim=0).to(dtype=torch.float)
                                p_lab_inp = torch.cat(p_lab, dim=0).to(dtype=torch.int)
                            elif len(p) == 1:
                                p_inp = p[0].to(dtype=torch.float)
                                p_lab_inp = p_lab[0].to(dtype=torch.int)
                            else:
                                raise Exception('Cannot be in the subloop for processing prompts that do not exist.')
                            #NOTE: Since we have transposed the slice extracted from the input image domain for the model domain input image, 
                            # when mapping to the model domain the corresponding coord will still have to undergo a rescaling corresponding to the scale factors 
                            # computed using the transposed image shape.
                            
                            # mapped_coords = self.apply_coords(p_inp, tuple(self.input_dom_shapes[ax][::-1]), self.app_params['image_size'])
                            mapped_coords = self.sam2_transforms.transform_coords(p_inp, normalize=True, orig_hw=tuple(self.input_dom_shapes[ax][::-1]))
                            if self.prop_freeform_uniformly:
                                #Turning the mapped coordinates into a batchwise set of sets, i.e. unsqueeze into BN2 where N=num_points, B=1. 
                                # We will not yet distribute the free-form prompts across instances, as it is inconclusive as to the quantity of instances being 
                                # segmented prior to reaching the prediction step.
                                self.model_prompts_storage_dict[ax][slice_idx][ptype] = mapped_coords.unsqueeze(dim=0).to(dtype=torch.float) 
                                self.model_prompts_storage_dict[ax][slice_idx][f'{ptype}_labels'] = p_lab_inp.unsqueeze(dim=0).to(dtype=torch.int)
                                assert torch.all(abs(mapped_coords[:,0]) <= self.app_params['image_size'][0]) and torch.all(abs(mapped_coords[:,1]) <= self.app_params['image_size'][1])
                            else:
                                NotImplementedError('No definitive strategy of splitting the free-form prompts, e.g. clicks and scribbles into different instances')
                            
                        elif ptype == 'bboxes':
                            #For bbox we concat as in the demo, we will for now pass through the labels too even though they won't be used for passing into the prompt encoder.
                            p, p_lab = self.orig_prompts_storage_dict[ax][slice_idx][ptype], self.orig_prompts_storage_dict[ax][slice_idx][f'{ptype}_labels']  
                            if len(p) > 1:
                                p_inp = torch.cat(p, dim=0).to(dtype=torch.float)
                                p_lab_inp = torch.stack(p_lab, dim=0) #We use stack, as we will distribute this across the batch-dim, "separate instances".
                                mapped_coords = self.sam2_transforms.transform_boxes(p_inp, normalize=True, orig_hw=self.input_dom_shapes[ax][::-1])
                                self.model_prompts_storage_dict[ax][slice_idx][ptype] = mapped_coords.to(dtype=torch.float) #Just in case. 
                                self.model_prompts_storage_dict[ax][slice_idx][f'{ptype}_labels'] = p_lab_inp.to(dtype=torch.int) #Just in case
                            elif len(p) == 1:
                                p_inp = p[0].to(dtype=torch.float)
                                p_lab_inp = torch.stack(p_lab, dim=0) 
                                mapped_coords = self.sam2_transforms.transform_boxes(p_inp, normalize=True, orig_hw=self.input_dom_shapes[ax][::-1])
                                self.model_prompts_storage_dict[ax][slice_idx][ptype] = mapped_coords.to(dtype=torch.float) #Just in case.
                                self.model_prompts_storage_dict[ax][slice_idx][f'{ptype}_labels'] = p_lab_inp.to(dtype=torch.int) #Just in case
                            else:
                                raise Exception('Cannot be in the subloop for processing prompts that do not exist.')
                            assert torch.all(abs(mapped_coords[:,:,0]) <= self.app_params['image_size'][0]) and torch.all(abs(mapped_coords[:,:,1]) <= self.app_params['image_size'][1])
    

# ####################################################################################
# #Functions for making the prediction given the input data which has been mapped to model domain.
    @torch.no_grad()
    def binary_predict(self, init_bool:bool, infer_slices: dict):
        #Function which takes the set of altered slices, the stored input information, and iterates through performing the predictions & updating the memory bank for
        #future iterations. It returns the dictionary of prob maps and dict of binary maps for each slice along each axial dimension as outputs.
        
        
        for ax in self.app_params['image_axes']:
            #We perform inference by axis (NOTE: for now since we have restricted this to being only a singular axis, any fusion across different axes is not implemented)
            if init_bool:
                self.internal_lowres_mask_storage[ax] = {slice_idx:None for slice_idx in range(len(self.image_features_dict[ax]))}
                self.internal_discrete_output_mask_storage[ax] = {slice_idx:None for slice_idx in range(len(self.image_features_dict[ax]))}
                self.internal_prob_output_mask_storage[ax] = {slice_idx:None for slice_idx in range(len(self.image_features_dict[ax]))}
                #Initialising the set of discrete and probabilistic output masks which will be used for constructing the output volumes. We will use this to store the
                #results such that when editing iterations occur, it can be altered at the corresponding slice and then re-merged.

            #Now we actually perform inference!
            for slice_idx in infer_slices[ax]:

                slice_ps = self.model_prompts_storage_dict[ax][slice_idx]
                #Finding the available prompts, we can use the fact that a non available prompt is empty because we pre-filtered the background bboxes. 
                # Therefore, no ambiguity here. A bounding box that is available is indeed available (we already pre-deleted the background bboxes...)
                avail_ps = [k for k in self.permitted_prompts if slice_ps[k] != [] and slice_ps[f'{k}_labels'] != []]
                
                if avail_ps == []:
                    #No prompt info provided.. this is treated like autoseg.
                    if self.autoseg_infer:
                        #In the case where we configure it to actually attempt to perform autoseg inference without any prompts, and then to use that for forward 
                        # propagation, we assume multi-mask to be used by default and hardcode this in spite of whatever the toggle may be (that pertains to cases where
                        # prompts were provided!). It is highly unlikely for this strategy to be effective in any capacity.
                        logits_outputs, _, lowres_masks = self.binary_slice_predict(self.image_features_dict[ax][slice_idx],
                                                                                    tuple(self.input_dom_shapes[ax][::-1].tolist()),
                                                                                    None,
                                                                                    None,
                                                                                    None,
                                                                                    None,
                                                                                    self.internal_lowres_mask_storage[ax][slice_idx],
                                                                                    True,
                                                                                    True)
                        #NOTE: Since we have transposed the slice extracted from the input image domain, mapping the predictions back to input domain will first 
                        # require transposing the shape of the slice, performing the inverse mapping wrt rescaling etc, transposition to back to input space, then
                        #combining all the slices together.

                        #Storing the lowres mask in memory. We follow the demo and reinsert this as the logits map.
                        self.internal_lowres_mask_storage[ax][slice_idx] = lowres_masks
                        #We keep the actual upsampled logits prediction separate by following the convention in the demo to use the lowres mask for forward propagation.
                        prob_outputs = torch.sigmoid(logits_outputs)
                        discrete_outputs = (prob_outputs > 0.5).long()
                    else:
                        #In the case where we do not actually perform autoseg inference, but just "skip over" and also pass a NoneType for future iterations such that
                        #the inference is not conditioned on a potentially sparse mask, but rather as though it is starting fresh.

                        #In this case, we return a tensor of -1000 for the output as this will eval to 0s at the floating point precision for the prob map under 
                        # sigmoid. Internally store a different mask variable, a Nonetype (i.e., it will treat that first interaction instance as an init..)
                        logits_outputs, lowres_masks = -1000 * torch.ones([1,1] + self.input_dom_shapes[ax][::-1].tolist()) , None
                        if not torch.all(torch.sigmoid(logits_outputs) == 0):
                            raise Exception('Error with the strategy for generating p = 0 maps.')
                        self.internal_lowres_mask_storage[ax][slice_idx] = lowres_masks  
                        #We keep these two separate by following the convention in the demo to use the lowres map for forward propagation.
                        prob_outputs = torch.sigmoid(logits_outputs).to(device=self.infer_device)
                        discrete_outputs = (prob_outputs > 0.5).long()
                else:
                    #In this case we have prompts, we split our next operations between points & scribbles, and bboxes (as we treat scribbles as sets of points)
                    # 
                    # First we do bboxes, as the potential constraint of requiring to propagate the set of prompts across each instance requires knowledge of the quantity
                    # of instances.
                      
                    if slice_ps['bboxes'] == [] or slice_ps['bboxes_labels'] == []:
                        input_bboxes_coords = None
                        input_bboxes_lbs = None 
                        multi_box_bool = None 
                    else:
                        assert slice_ps['bboxes'].shape[0] == slice_ps['bboxes_labels'].shape[0]
                        #In this case we have bbox prompts that are valid.
                        if slice_ps['bboxes_labels'].shape[0] == 1:
                            #In this case we only have one box in this slice.
                            multi_box_bool = False 
                        else:
                            #In this case we have more than one box in this slice. We will use this variable for unpacking the outputs appropriately.
                            multi_box_bool = True 
                            warnings.warn('The probabilistic map output for a multi-box process will be fused in a very naive manner, by taking the maximum of the probability maps voxelwise, note that the discrete prediction is however fused in the exact same manner as the demo (by discretising separately and then finding the union).')
                        input_bboxes_coords = slice_ps['bboxes'].to(device=self.infer_device)
                        input_bboxes_lbs = slice_ps['bboxes_labels'].to(device=self.infer_device)
                        assert input_bboxes_coords.shape[0] == input_bboxes_lbs.shape[0]                       
                    
                    if (slice_ps['points'] == [] or slice_ps['points_labels'] == []) and (slice_ps['scribbles'] == [] or slice_ps['scribbles_labels'] == []):
                        input_points_coords = None
                        input_points_lbs = None 
                    else:
                        #At least one of the scribbles and points is valid
                        point_coors, point_lbs = [], []
                        for p in ['points', 'scribbles']:
                            if (slice_ps[p] != [] and slice_ps[f'{p}_labels'] != []):
                                point_coors.append(slice_ps[p])
                                point_lbs.append(slice_ps[f'{p}_labels'])
                        assert len(point_coors) != 0

                        #The points need to be in B x N_point x N_dim structure and lbs need to be in B x N_point structure, where B has been determined according 
                        # to the bbox quantity IF there are bboxes. Otherwise, it should be 1.
                        input_points_coords = torch.cat(point_coors, dim=1) #First we stack the clicks and scribbles' points together.
                        input_points_lbs = torch.cat(point_lbs, dim=1)
                        #Now we distribute these points across the different instances as indicated by the bbox quantity.
                        if input_bboxes_coords is None and input_bboxes_lbs is None:                            
                            input_points_coords = input_points_coords.to(device=self.infer_device)
                            input_points_lbs = input_points_lbs.to(device=self.infer_device)
                            assert input_points_coords.shape[0] == input_points_lbs.shape[0]

                        elif input_bboxes_coords is not None and input_bboxes_lbs is not None:
                            input_points_coords = input_points_coords.repeat(input_bboxes_coords.shape[0],1,1).to(device=self.infer_device)
                            input_points_lbs = input_points_lbs.repeat(input_bboxes_coords.shape[0], 1).to(device=self.infer_device)
                            assert input_points_coords.shape[0] == input_bboxes_coords.shape[0]
                            assert input_points_lbs.shape[0] == input_bboxes_lbs.shape[0]
                        
                        else:
                            raise Exception('What happened here...? Only one of the bbox coords and labels were NoneType?')
                        

                    if (input_points_coords is None or input_points_lbs is None) and (input_bboxes_coords is None or input_bboxes_lbs is None):
                        raise Exception('Cannot have Nonetype for both points and bbox if we are in the subloop for performing inference with prompts.')                    
                    
                    #Logic which controls whether the multi-mask is configured, can optionally always be true, and can also be something that needs to be determined from the
                    #prompts provided.
                    if self.multimask_output_always:
                        multi_ambig_mask_bool = True 
                    else:
                        # NotImplementedError('No strategy yet defined for determining what a sufficient quantity of prompts is for not requiring disambiguation')   
                        warnings.warn('Highly experimental, determining whether or not to use the multi-mask strategy for disambiguation with a very simple heuristic.')
                        if (input_bboxes_coords is not None) or (input_points_coords is not None and input_points_coords.shape[1] > 1):
                            multi_ambig_mask_bool = False #We follow the logic used in the demo, where bbox is classed as an unambiguous prompt, and also instances with
                            # > 1 click/point is unambiguous also (this might not be necessarily a good assumption)
                        else: 
                            multi_ambig_mask_bool = True
                            #If no bbox and only 1 point, then use the multi-mask, as with the demo's guidance. 

                    #NOTE: Since we have transposed the slice extracted from the input image domain, mapping the predictions back to input domain will first 
                    # require reversing the ordered tuple denoting the shape of the slice, performing the inverse mapping wrt rescaling etc, transposition to back to input space, then
                    #combining all the slices together.
                    logits_outputs, _, lowres_masks = self.binary_slice_predict(self.image_features_dict[ax][slice_idx], 
                                                                                self.input_dom_shapes[ax][::-1].tolist(),
                                                                                input_points_coords,
                                                                                input_points_lbs, 
                                                                                input_bboxes_coords,
                                                                                input_bboxes_lbs, 
                                                                                self.internal_lowres_mask_storage[ax][slice_idx], 
                                                                                multi_ambig_mask_bool,
                                                                                True)
                    #NOTE: Very naive fusion strategy coming up for the probabilistic map, we take the max (because they're all supposed to be foreground of the same class). 
                    # For the discrete map we just take the union of the discrete maps generated for each bbox.
                    
                    #Here we store the lowres mask internally, in the same capacity as done with the demo.

                    self.internal_lowres_mask_storage[ax][slice_idx] = lowres_masks
                    
                    if multi_box_bool is not None and not multi_box_bool:
                        #Single box, pretty straight forward to evaluate this.
                        prob_outputs = torch.sigmoid(logits_outputs)
                        discrete_outputs = (prob_outputs > 0.5).long()
                    elif multi_box_bool is not None and multi_box_bool:
                        mask_dim = 0
                        if not logits_outputs.shape[mask_dim] > 1:
                            raise Exception(f'We implemented this wrong, the mask dimension from output indicates that 1 or fewer bboxes were used but we are in the handling for multiple bboxes')
                        #Multiple box, we take a naive approach for handling the probabilistic map output, we take max over all channels as it is a single foreground!
                        box_sep_prob_outputs = torch.sigmoid(logits_outputs)
                        discrete_outputs = (box_sep_prob_outputs > 0.5).long()
                        #We reduce over the 0th dimension corresponding to the quantity of prompts which are treated as distinct object instances (i.e. for each bbox).
                        discrete_outputs = (discrete_outputs.sum(dim=mask_dim, keepdim=True) > 0).long() #We sum over the mask dim, then binarise as we assume each instance is
                        #an instance of the given foreground class (and we are performing semantic segmentation)
                        
                        #Now we aggregate the probability map we want to output.
                        prob_outputs = box_sep_prob_outputs.max(dim=mask_dim, keepdim=True)[0] 
                        #.max returns a tuple of the tensor and the tensor of indices along that channel and where the max occurs (i.e. the argmax) 
                    elif multi_box_bool is None:
                        if slice_ps['bboxes'] != [] or slice_ps['bboxes_labels'] != []:
                            raise Exception('Should not have flagged box as being NoneType if there were boxes.')
                        #For non-box prompt types, we have little to worry about, it doesn't treat each prompt as a separate instance..
                        prob_outputs = torch.sigmoid(logits_outputs)
                        discrete_outputs = (prob_outputs > 0.5).long()
            
                #Storing the output maps, first we check that the shapes are consistent with what is required, ESPECIALLY for the channel dimensions:
                #we reverse the list because the input dom shapes are extracted prior to the transposition required for mapping from RAS to y,x cv2 coordinates.
                if list(prob_outputs.shape) != [1, 1] + self.input_dom_shapes[ax][::-1].tolist() or list(discrete_outputs.shape) != [1, 1] + self.input_dom_shapes[ax][::-1].tolist():
                    raise Exception('The structure of the output discrete map, and output probability map (prior to undoing the transposition operation) should be identical to the input image spatial size.')
                self.internal_prob_output_mask_storage[ax][slice_idx] = prob_outputs[0,0,...]
                self.internal_discrete_output_mask_storage[ax][slice_idx] = discrete_outputs[0,0,...] 
                
                #Plotting the output as a sanity check.
                if slice_idx == self.sanity_slice_check:
                    sanity_check_output(prob_outputs[0,0,...], 'prob', self.orig_prompts_storage_dict[ax][slice_idx], False)
                    sanity_check_output(discrete_outputs[0,0,...], 'discrete', self.orig_prompts_storage_dict[ax][slice_idx], False)

            if any([val is None for val in self.internal_discrete_output_mask_storage[ax].values()]):
                raise Exception('We should not have a NoneType for any slice after performing inference with respect to the internal discrete mask storage')
            if any([val is None for val in self.internal_prob_output_mask_storage[ax].values()]):
                raise Exception('We should not have a NoneType for any slice after performing inference with respect to the internal probability mask storage')


    @torch.no_grad() #Function is taken from the SAM2 demo prediction function.
    def binary_slice_predict(
        self,
        image_features: dict[torch.Tensor],
        slice_dimensions: tuple,
        point_coords: torch.Tensor | None,
        point_labels: torch.Tensor | None,
        boxes_coords: torch.Tensor | None,
        boxes_labels: torch.Tensor | None,
        mask_input: torch.Tensor | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using SAM2Transforms.

        Arguments:

          image_features (dict[torch.tensor]): A dictionary containing the image features
            of the given image slice that were extracted by the ViT encoder, required 
            for performing inference.

          slice_dimensions (tuple): A 2-tuple containing the original image resolution 
            for the given slice for post-processing.

          point_coords (torch.Tensor or None): A BxNx2 tensor of point prompts to the
            model. Each point within each batch is in (X,Y) format in pixel units.
                
                Expected to be on the same device as the feature embeddings.

          point_labels (torch.Tensor or None): A BxN tensor of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
                
                Expected to be on the same device as the feature embeddings.

          boxes_coords (torch.Tensor or None): A Bx 2 x 2 tensor representing a set of box prompts
            where for each box prompt to the model it has been transformed from XYXY format to the 
            format required for inference:
            
            X1 Y1
            X2 Y2.

            Expected to be on the same device as the feature embeddings.

          boxels_labels (torch.Tensor or None): A B x 1 tensor giving the label of the bbox (for checking).

            Expected to be on the same device as the feature embeddings.
          
          mask_input (torch.Tensor or None): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. If it previously was split by
            instances wrt bboxes, then it can have form Bx1xHxW with B > 1 where
            for SAM, H=W=256. If the mask was previously initialised without instance-based info,
            e.g., for bbox-free prompt-based initialised masks or autoseg potentially, then B = 1.
            
          multimask_output (bool): If true, the model will return three masks.
            One for each instance (batch-dimension).
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.


        Returns:
          (torch.Tensor): The output mask in Bx1xHxW format, where B is the
            number of instances, and (H, W) is the original image size.
          (torch.Tensor): A tensor of shape Bx1  containing the model's
            prediction for the quality of the best mask for each instance.
          (torch.Tensor): A tensor of shape Bx1xHxW, where B is the number
            of instances, masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """

        if point_coords is not None:
            if not point_coords.device == self.infer_device:
                point_coords = point_coords.to(device=self.infer_device)
            if not point_labels.device == self.infer_device:
                point_labels = point_labels.to(device=self.infer_device)
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes_coords is not None:
            # box_coords = boxes.reshape(-1, 2, 2)
            if not boxes_coords.device == self.infer_device:
                boxes_coords = boxes_coords.to(device=self.infer_device)
            if not boxes_labels.device == self.infer_device:
                boxes_labels = boxes_labels.to(device=self.infer_device) 

            if not all([i == 1 for i in boxes_labels.flatten()]):
                raise Exception('Should have removed a non-foreground bbox earlier!')
            #Now reassign the bbox labels now that we have actually checked that there are no background bboxes.
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes_coords.device)
            box_labels = box_labels.repeat(boxes_coords.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([boxes_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (boxes_coords, box_labels)
     
        if mask_input is not None and mask_input.shape[0] > 1: #In this case the mask has been pre-initialised with a set of bboxes to obtain instance-level masks.
            if concat_points[0].shape[0] != mask_input.shape[0]: 
                raise Exception('The instance count/batch-dim cannot be inconsistent between the prompts and the masks post-bbox initialisation/instance-level initialisation.')
        elif mask_input is not None and mask_input.shape[0] == 1:
            if concat_points[0].shape[0] != mask_input.shape[0]:
                #In this case, we had a singular previous mask, and if we have a batch-dim/instance count > 1, then we need to distribute the mask. This can happen if
                #someone initialises the segmentation process with a non bbox (highly unlikely but still possible), or potentially with an initialisation mask.
                mask_input = mask_input.repeat(concat_points[0].shape[0],1,1,1)
        elif mask_input is None:
            pass #We just make this explicit for readability.

        #NOTE: The boxes are not non-existent, they've just encoded them under the points through a combination of dense positional encoding and labels specifically
        # for denoting the extrema of the bbox!
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = [
            feat_level[img_idx].unsqueeze(0) #What does this actually do...? Well, in image-batch (not instance batch mode) mode the img idx is as described. So effectively extracting a given
            #image-batchwise feature tensor would reduce the dimension, if keepdim=False. We have not touched the implementation chosen by the demo in spite of the fact that
            #we are working with a single this will just ensure that the tensor remains a 4 dimensional tensor.
            for feat_level in image_features["high_res_feats"]
        ]
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=image_features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        # Upscale the masks to the original image resolution
        masks = self.sam2_transforms.postprocess_masks(
            low_res_masks, slice_dimensions #We do not bother with the use of img_idx here, we just need the dimensions of the slice, we presume that the same exact image
            # is being provided across all cases in the batchwise inference, and that the "batch" is just instance level segmentation.
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        
        if concat_points is None: #We extract the quantity of instances which are being outputted as masks, this must match the batchdim/instance size of the input.
            b_dim = 1
        else:
            b_dim = concat_points[0].shape[0]
        #We write our assertions separately just for clarity.
        assert masks.shape[0] == b_dim
        assert iou_predictions.shape[0] == b_dim
        assert low_res_masks.shape[0] == b_dim

        if multimask_output:
            #If multi-mask output is used then extract the optimal mask. We have a B3HW structure.
            max_values, max_index = torch.max(iou_predictions, dim=1) 
            #This is taking the maximum across the mask channel, not the batch! So the multi-instance aspect is not a problem!
            
            #Restructure the iou predictions such that it is B x 1.
            iou_predictions = max_values.unsqueeze(1)
            #We have to restructure the indices to be able to index the correct mask corresponding to the max iou for each batch item 
            low_res_masks = low_res_masks[torch.arange(b_dim), max_index] # Shape: [B, H, W]
            low_res_masks = low_res_masks.unsqueeze(1)
            masks = masks[torch.arange(b_dim), max_index] # Shape: [B, H, W]
            masks = masks.unsqueeze(1)

        #We write our assertions separately just for clarity.
        assert iou_predictions.shape[1] == 1 #We are only taking the best, or a singular IOU prediction forward for output.
        assert low_res_masks.shape[1] == 1 #We are only taking a singular mask forward for each instance/batch item.
        assert masks.shape[1] == 1 #We are only taking a singular mask forward for each instance.
    
        
    
        if not return_logits:
            masks = masks > self.mask_threshold_logits
        
        return masks, iou_predictions, low_res_masks



# ############################################################

    def binary_merge_slices(self):
        """
        Slice merging steps:

            - Combine inferred slices into one volume while also undoing the transposition according to the axes (swapping the channels) such that the output 
            matches the RAS structure of the input image.
            
                This is done due to the fact that the input image in RAS would not be aligned with the coordinates used by cv2.
        """
        #We assert that the quantity of axes once again must be 1, since there is not an obvious way to merge slices otherwise with a concatenation.

        if not len(self.app_params['image_axes']) == 1:
            raise Exception('Cannot merge together slices in multiple axes with a simple concatenation')
        

        for ax in self.app_params['image_axes']:
            merged_discrete = torch.cat([(mask.T).unsqueeze(dim=ax) for mask in self.internal_discrete_output_mask_storage[ax].values()], dim=ax)
            merged_prob = torch.cat([(mask.T).unsqueeze(dim=ax) for mask in self.internal_prob_output_mask_storage[ax].values()], dim=ax)
            
            assert self.orig_im_shape == tuple(merged_discrete.shape)
            assert self.orig_im_shape == tuple(merged_prob.shape)

        #Now we channel-split the probability map by class.
        prob_map_list = []
        for label in self.configs_labels_dict.keys():
            if label.title() == 'Background':
                prob_map_list.append(1 - merged_prob)
            else:
                prob_map_list.append(merged_prob)
        merged_prob = torch.stack(prob_map_list, dim=0)
        merged_discrete = merged_discrete.unsqueeze(dim=0)
        
        assert merged_discrete.ndim == 4
        assert merged_prob.ndim == 4
        assert merged_prob.shape[0] == len(self.configs_labels_dict)

        return merged_discrete, merged_prob #merged_prob.unsqueeze(dim=0)

    def __call__(self, request:dict):

        if len(request['config_labels_dict']) == 2:
            class_type = 'binary'
        elif len(request['config_labels_dict']) > 2:
            class_type = 'multi'
            raise NotImplementedError 
        else:
            raise Exception('Should not have received less than two class labels at minimum')
        
        #We create a duplicate so we can transform the data from metatensor format to the torch tensor format compatible with the inference script.
        modif_request = copy.deepcopy(request) 

        app = self.infer_apps[modif_request['model']][f'{class_type}_predict']

        #Setting the configs label dictionary for this inference request.
        self.configs_labels_dict = modif_request['config_labels_dict']


        pred, probs_tensor, affine = app(request=modif_request)




        assert probs_tensor.shape[1:] == request['image']['metatensor'].shape[1:]
        assert pred.shape[1:] == request['image']['metatensor'].shape[1:] 
        assert torch.all(affine == request['image']['metatensor'].meta['affine'])
        assert isinstance(probs_tensor, torch.Tensor) 
        assert isinstance(pred, torch.Tensor)
        assert isinstance(affine, torch.Tensor)

        output = {
            'probs':{
                'metatensor':probs_tensor.to(device='cpu'),
                'meta_dict':{'affine': affine.to(device='cpu')}
            },
            'pred':{
                'metatensor':pred.to(device='cpu'),
                'meta_dict':{'affine': affine.to(device='cpu')}
            },
        }
        return output 
    
if __name__ == '__main__':
   
    infer_app = InferApp(
        {'dataset_name':'BraTS2021',
        'dataset_modality':'MRI'}, torch.device('cuda'))

    infer_app.app_configs()

    from monai.transforms import LoadImaged, Orientationd, EnsureChannelFirstd, Compose 
    import nibabel as nib 

    input_dict = {'image':'/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_binarised/imagesTs/BraTS2021_00266.nii.gz'}
    load_and_transf = Compose([LoadImaged(keys=['image']), EnsureChannelFirstd(keys=['image']), Orientationd(keys=['image'], axcodes='RAS')])

    input_metatensor = load_and_transf(input_dict)
    request = {
        'image':{
            'metatensor': input_metatensor['image'],
            'meta_dict':{'affine':input_metatensor['image'].affine}
        },
        # 'model':'IS_interactive_init',
        'model': 'IS_autoseg',
        'config_labels_dict':{'background':0, 'tumor':1},
        'im':
        {'Automatic Init': None},
        # {'Interactive Init':{
        #     'interaction_torch_format': {
        #         'interactions': {
        #             'points': [torch.tensor([[40, 103, 43]]), torch.tensor([[62, 62, 39]]), torch.tensor([[61, 62, 39]]), torch.tensor([[80,35,39]]), torch.tensor([[81,35,39]])], #None
        #             'scribbles': [torch.tensor([[63,62,39], [64,62,39],[65,62,39]])],#, torch.tensor([[73,62,39], [74,62,39],[75,62,39]])], 
        #             #This second scribble is a fugazi but intended just for sanity checking the mapping.
        #             'bboxes': [torch.Tensor([[56,30,17, 92, 76, 51]]).to(dtype=torch.int64)], #, torch.Tensor([[93,80,30, 105, 100, 51]]).to(dtype=torch.int64)]  #None 
        #             },#This second box is a fugazi but intended just for sanity checking that the multi-box method works. 17-51 should be the real bounds.
        #         'interactions_labels': {
        #             'points_labels': [torch.tensor([0]), torch.tensor([1]), torch.tensor([1]), torch.tensor([0]), torch.tensor([0])], #None,
        #             'scribbles_labels':[torch.tensor([1])],#, torch.tensor([0])],  
        #             'bboxes_labels': [torch.Tensor([1]).to(dtype=torch.int64)]#, torch.Tensor([1]).to(dtype=torch.int64)] #None
        #             },
        #         }
        #     }
        # },
    }
    output = infer_app(request)
    # print('halt')



    request2 = {
        'image':{
            'metatensor': input_metatensor['image'],
            'meta_dict':{'affine':input_metatensor['image'].affine}
        },
        'model':'IS_interactive_edit',
        'config_labels_dict':{'background':0, 'tumor':1},
        'im':
        {'Interactive Edit Iter 1':{
            'interaction_torch_format': {
                'interactions': {
                    'points': [torch.tensor([[62,62,39]]), torch.tensor([[41, 103, 69]]), torch.tensor([[62, 62, 57]]), torch.tensor([[61, 62, 57]])], #None
                    'scribbles': [torch.tensor([[63,62,39], [64,62,39],[65,62,39]])],#, torch.tensor([[73,62,57], [74,62,57],[75,62,57]])], 
                    #This second scribble is a fugazi but intended just for sanity checking the mapping.
                    'bboxes': [torch.Tensor([[56,30,17, 92, 76, 39]]).to(dtype=torch.int64), torch.Tensor([[93,80,30, 105, 100, 51]]).to(dtype=torch.int64)]  #None 
                    },#This second box is a fugazi but intended just for sanity checking that the multi-box method works. 17-51 should be the real bounds.
                'interactions_labels': {
                    'points_labels': [torch.tensor([1]), torch.tensor([0]), torch.tensor([1]), torch.tensor([1])], #None,#[torch.tensor([0]), torch.tensor([1])], 
                    'scribbles_labels':[torch.tensor([1])],#, torch.tensor([0])],  
                    'bboxes_labels': [torch.Tensor([1]).to(dtype=torch.int64), torch.Tensor([1]).to(dtype=torch.int64)] #None
                    }
                },
            }
        },
    }

    output2 = infer_app(request2)
    print('halt')