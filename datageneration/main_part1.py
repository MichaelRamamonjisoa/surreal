import sys
import os
import random
import math
import bpy
import numpy as np
#from os import getenv
#from os import remove
from os.path import join, dirname, realpath, exists
from mathutils import Matrix, Vector, Quaternion, Euler
#from glob import glob
from random import choice
from pickle import load
from bpy_extras.object_utils import world_to_camera_view as world2cam

sys.path.insert(0, ".")

def select_rand_file_from_txtlist(txtFilename):
    with open(txtFilename, 'r') as f:
        list_fn = [line.split('\n')[0] for line in f]
        
    chosen_filename = choice(list_fn)
    return(chosen_filename)

def build_background(room='bedroom', walls=['N','W'], width=25, length=15, height=2.4):
    
    #Build walls and floor
    bpy.ops.mesh.archimesh_room()
    height_2 = height + 3
    bpy.data.objects["Room"].RoomGenerator[0].wall_num = 3    
    bpy.data.objects["Room"].RoomGenerator[0].wall_width = 0.09    
    bpy.data.objects["Room"].RoomGenerator[0].walls[0].w = width
    bpy.data.objects["Room"].RoomGenerator[0].walls[1].w = length
    bpy.data.objects["Room"].RoomGenerator[0].walls[2].w = -width             
    bpy.data.objects["Room"].RoomGenerator[0].merge = True
    bpy.data.objects["Room"].RoomGenerator[0].walls[0].h = '3'
    bpy.data.objects["Room"].RoomGenerator[0].walls[1].h = '3'
    bpy.data.objects["Room"].RoomGenerator[0].walls[2].h = '3'
    bpy.data.objects["Room"].RoomGenerator[0].room_height = height_2
    
#    bpy.data.objects["Room"].RoomGenerator[0].floor = True 
#    bpy.data.objects["Room"].RoomGenerator[0].ceiling = True 
    bpy.data.objects["Room"].location = (-width/2,-length/2,0)
    
    
    the_folder = join(os.path.expanduser('~'), 'Documents', 'MVA', 'datageneration')
    tex_folder = join(the_folder,'Texture_Material','dtd')
                                                  
    if room=='bedroom':        
        # load bed
#        load_IKEA_dtd('bed',(2,2,0.0),1,0)
        load_IKEA_dtd('bed',(np.random.uniform(-4.0,4.0),np.random.uniform(-4.0,4.0),0.0),1,np.random.uniform(-180,179))
        # load bookcase
#        load_IKEA_dtd('bookcase',(2.0,3.0,0.0),1,0)
        load_IKEA_dtd('bookcase',(np.random.uniform(-4.0,4.0),np.random.uniform(-4.0,4.0),0.0),1,np.random.uniform(-180,179))
        # load desk
#        load_IKEA_dtd('desk',(2.0,-2.0,0.0),1,0)
        load_IKEA_dtd('desk',(np.random.uniform(-4.0,4.0),np.random.uniform(-4.0,4.0),0.0),1,np.random.uniform(-180,179))
        # load chair 
#        load_IKEA_dtd('chair',(0.0,2.0,0.0),0.7,np.random.uniform(-180,179))
        load_IKEA_dtd('chair',(np.random.uniform(-4.0,4.0),np.random.uniform(-4.0,4.0),0.0),0.7,np.random.uniform(-180,179))
        # load wardrobe
#        load_IKEA_dtd('wardrobe',(4.0,2.0,0.0),1,0)
        load_IKEA_dtd('wardrobe',(np.random.uniform(-4.0,4.0),np.random.uniform(-4.0,4.0),0.0),1,np.random.uniform(-180,179))
        
    if room=='living':
        # load sofa
#        load_IKEA_dtd('sofa',(2,2,0.0),1,0)
        load_IKEA_dtd('sofa',(np.random.uniform(-4.0,4.0),np.random.uniform(-4.0,4.0),0.0),1,np.random.uniform(-180,179))
        # load bookcase
#        load_IKEA_dtd('bookcase',(2.0,3.0,0.0),1,0)
        load_IKEA_dtd('bookcase',(np.random.uniform(-4.0,4.0),np.random.uniform(-4.0,4.0),0.0),1,np.random.uniform(-180,179))
        # load desk
#        load_IKEA_dtd('desk',(2.0,-2.0,0.0),1,0)
        load_IKEA_dtd('desk',(np.random.uniform(-4.0,4.0),np.random.uniform(-4.0,4.0),0.0),1,np.random.uniform(-180,179))
        # load chair 
        load_IKEA_dtd('chair',(np.random.uniform(-4.0,4.0),np.random.uniform(-4.0,4.0),0.0),0.7,np.random.uniform(-180,179))
        load_IKEA_dtd('chair',(np.random.uniform(-4.0,4.0),np.random.uniform(-4.0,4.0),0.0),0.7,np.random.uniform(-180,179))
        load_IKEA_dtd('chair',(np.random.uniform(-4.0,4.0),np.random.uniform(-4.0,4.0),0.0),0.7,np.random.uniform(-180,179))
        load_IKEA_dtd('chair',(np.random.uniform(-4.0,4.0),np.random.uniform(-4.0,4.0),0.0),0.7,np.random.uniform(-180,179))
        # load table
        load_IKEA_dtd('table',(4.0,2.0,0.0),1.6,0)
        
    #group background objects together
    if not 'bg_group' in bpy.data.groups:
        objects = bpy.context.scene.objects
        list_objects = [ob for ob in objects if ob.layers[0] and ob.name!='Camera']
        group = bpy.data.groups.new("bg_group")
        for ob in list_objects:
            group.objects.link(ob)
            
    # Random light position and intensity
    #remove 'Lamp' object
    objs = bpy.data.objects
    objs.remove(objs["Lamp"], True)    
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.mesh.primitive_uv_sphere_add(segments=8,ring_count=8)
    light_sphere = bpy.data.objects["Sphere"]
    mat_light = bpy.data.materials.new(name="LightMaterial")
    bpy.ops.object.material_slot_add()
    light_sphere.material_slots[0].material = mat_light
    
    #make node tree
    bpy.data.materials[mat_light.name].use_nodes = True          
    light_tree = mat_light.node_tree
    
    for n in light_tree.nodes:
        light_tree.nodes.remove(n)    
        
    emission = light_tree.nodes.new("ShaderNodeEmission")
    light_out = light_tree.nodes.new("ShaderNodeOutputMaterial")
    
    light_tree.links.new(emission.outputs[0],light_out.inputs[0])    
    
    light_pos = ((np.random.rand(1)-0.5)*width*0.7,(np.random.rand(1)-0.5)*length*0.7,(np.random.rand(1)-0.5)*height)

#    light_angle = (np.random.rand(1)*360-180,np.random.rand(1)*360-180,np.random.rand(1)*360-180)
    light_size = np.random.rand(1)
    light_sphere.location = light_pos
#    light_plane.rotation_euler = light_angle
    light_sphere.scale = light_size*(1,1,1)
    light_intensity = np.random.rand(1)*100
    emission.inputs[1].default_value = light_intensity   

    #make light plane invisible
    light_sphere.cycles_visibility.camera = False
    light_sphere.cycles_visibility.glossy = False
    light_sphere.cycles_visibility.transmission = False
    light_sphere.cycles_visibility.scatter = False
    light_sphere.cycles_visibility.shadow = False                 
            
    rot_x_neg90 = Matrix.Rotation(math.pi/2.0, 4, 'X')
    
    bpy.ops.object.select_all(action='DESELECT') 
    
    gen = (obj for obj in bpy.data.objects if obj.name!='Camera')
    for obj in gen:
        obj.matrix_world = rot_x_neg90 * obj.matrix_world            
    
    # Add materials to wall and floor
    # Walls
    for o in bpy.data.objects:
        o.select = False        
    obj = bpy.data.objects["Room"]
    obj.select = True
    bpy.context.scene.objects.active = obj            
    bpy.ops.object.material_slot_remove()
    mat_wall = bpy.data.materials.new(name="WallMaterial")
    bpy.ops.object.material_slot_add()
    
    mat_type = choice(['banded','lined', 'paisley', 'polka-dotted', 'striped', 'zigzagged'])
#    log_message('WallMaterialType: %s' % mat_type)
    wall_file = select_rand_file_from_txtlist(join(tex_folder,'for_surreal',mat_type+'.txt'))  
#    log_message('WallMaterial: %s' % wall_file)      
    addImg2Mat(join(tex_folder,'images',mat_type,wall_file),mat_wall.name,(0.2,0.2,0.2))          
    
    for o in bpy.data.objects:
        o.select = False       
    bpy.data.objects["Room"].select = True
    bpy.data.objects["Room"].RoomGenerator[0].floor = True    
    bpy.data.objects["Room"].select = False
    obj = bpy.data.objects["Floor"]
    obj.select = True
    bpy.context.scene.objects.active = obj            
    bpy.ops.object.material_slot_remove()
    mat_floor = bpy.data.materials.new(name="FloorMaterial")
    bpy.ops.object.material_slot_add()   
    
    floor_txtfile = join(the_folder,'Texture_Material', "floor.txt")    
    select_tex_floor = select_rand_file_from_txtlist(floor_txtfile)
    if select_tex_floor == 'marbled':
        floor_file = select_rand_file_from_txtlist(join(tex_folder,'for_surreal','marbled.txt')) 
        addImg2Mat(join(tex_folder,'images','marbled',floor_file),mat_floor.name,(0.25,0.25,0.25)) 
    else:
        floorImgPath = join(the_folder,'Texture_Material',select_tex_floor)
        addImg2Mat(floorImgPath,mat_floor.name,imgScale = (0.25,0.25,0.25))
        
    bpy.data.objects["Room"].material_slots[0].material = mat_wall             
    bpy.data.objects["Floor"].material_slots[0].material = mat_floor
#    ## Uncomment to add ceiling
#    bpy.data.objects["Room"].RoomGenerator[0].ceiling = False 
#    bpy.data.objects["Room"].RoomGenerator[0].ceiling = True
#    
#    for o in bpy.data.objects:
#        o.select = False        
#    obj = bpy.data.objects["Ceiling"]
#    obj.select = True
#    bpy.context.scene.objects.active = obj            
#    bpy.ops.object.material_slot_remove()
#    mat_ceil = bpy.data.materials.new(name="CeilMaterial")
#    bpy.ops.object.material_slot_add()   
#    
#    ceil_txtfile = join(the_folder,'Texture_Material', "floor.txt")    
#    select_tex_ceil = select_rand_file_from_txtlist(ceil_txtfile)
#    if select_tex_ceil == 'marbled':
#        ceil_file = select_rand_file_from_txtlist(join(tex_folder,'for_surreal','marbled.txt')) 
#        addImg2Mat(join(tex_folder,'images','marbled',ceil_file),mat_ceil.name,(0.25,0.25,0.25)) 
#    else:
#        floorImgPath = join(the_folder,'Texture_Material',select_tex_floor)
#        addImg2Mat(floorImgPath,mat_ceil.name,imgScale = (0.25,0.25,0.25))
#             
#    obj.material_slots[0].material = mat_ceil
    
    
       
    return light_pos, light_size, light_intensity

        
def sum_coords(tuple1,tuple2):
    n = len(tuple1)
    a = [0]*n
    for i in range(0,n):
        a[i] = tuple1[i] + tuple2[i]
    return(tuple(a))
    
def load_IKEA_dtd(item_category='bed', item_location=(0,0,0), item_scale = 1, item_z_angle = 0):
    
    #build collection list
    list_obj = []
    the_folder = join(os.path.expanduser('~'), 'Documents', 'MVA', 'datageneration')
    tex_folder = join(the_folder,'Texture_Material','dtd')
    
    IKEA_folder = join(the_folder,'3dobj_data', 'IKEA')
    with open(join(IKEA_folder, "all_obj.txt"), "r") as f:
        list_obj = ['IKEA_'+line.split('IKEA_')[1].split('\x1b')[0] for line in f if len(line.split('IKEA_'+item_category))==2]
         
    select_obj = choice(list_obj)
    file_loc = join(IKEA_folder, select_obj)
    imported_object = bpy.ops.import_scene.obj(filepath=file_loc,use_split_objects=False,use_split_groups=False,axis_forward='Z',axis_up='Y')
    obj = bpy.context.selected_objects[0]
    
    obj_name = obj.name
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')    
    (x,y,z) = (item_scale/(min([(obj.dimensions[0]+obj.dimensions[1]+obj.dimensions[2])/3,max(obj.dimensions)])))*obj.dimensions
    bpy.data.objects[obj_name].dimensions = (x,y,z)
    bpy.data.objects[obj_name].location = sum_coords(item_location,(0,0,y/2))
    
    (tx,ty,tz) = bpy.data.objects[obj_name].rotation_euler 
    bpy.data.objects[obj_name].rotation_euler = sum_coords((tx,ty,tz),(0,0,item_z_angle))
    
    #add random materials to IKEA objects from the Describable Texture Dataset
    
    for i,slot in enumerate(bpy.data.objects[obj_name].material_slots):
        mat = bpy.data.objects[obj_name].material_slots[i].material
        mat_type = select_rand_file_from_txtlist(join(tex_folder,'dtd_types.txt'))
        mat_file = select_rand_file_from_txtlist(join(tex_folder,'for_surreal',mat_type+'.txt'))        
        addImg2Mat(join(tex_folder,'images',mat_type,mat_file),mat.name,(1,1,1))                
        
    return(obj_name)

def addImg2Mat(imgFullPath,matName,imgScale = (0.5,0.5,0.5)):
    
    scene = bpy.data.scenes['Scene']
    scene.render.engine = 'CYCLES'
    bpy.data.materials[matName].use_nodes = True
    img = bpy.data.images.load(imgFullPath)
    
    mat_tree = bpy.data.materials[matName].node_tree
    
    # clear default nodes
    for n in mat_tree.nodes:
        mat_tree.nodes.remove(n)

    texImg = mat_tree.nodes.new("ShaderNodeTexImage")
    texImg.image = img
    
    bsdf = mat_tree.nodes.new("ShaderNodeBsdfDiffuse")
    mat_out = mat_tree.nodes.new("ShaderNodeOutputMaterial")
    
    uv = mat_tree.nodes.new('ShaderNodeTexCoord')
    mapp = mat_tree.nodes.new('ShaderNodeMapping')    
    mapp.vector_type = 'TEXTURE'
    mapp.scale = imgScale
       
    mat_tree.links.new(uv.outputs[2],mapp.inputs[0])
    mat_tree.links.new(mapp.outputs[0],texImg.inputs[0])
    mat_tree.links.new(texImg.outputs[0],bsdf.inputs[0])    
    mat_tree.links.new(bsdf.outputs[0], mat_out.inputs[0])

def mkdir_safe(directory):
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

def setState0():
    for ob in bpy.data.objects.values():
        ob.select=False
    bpy.context.scene.objects.active = None

sorted_parts = ['hips','leftUpLeg','rightUpLeg','spine','leftLeg','rightLeg',
                'spine1','leftFoot','rightFoot','spine2','leftToeBase','rightToeBase',
                'neck','leftShoulder','rightShoulder','head','leftArm','rightArm',
                'leftForeArm','rightForeArm','leftHand','rightHand','leftHandIndex1' ,'rightHandIndex1']
# order
part_match = {'root':'root', 'bone_00':'Pelvis', 'bone_01':'L_Hip', 'bone_02':'R_Hip',
              'bone_03':'Spine1', 'bone_04':'L_Knee', 'bone_05':'R_Knee', 'bone_06':'Spine2',
              'bone_07':'L_Ankle', 'bone_08':'R_Ankle', 'bone_09':'Spine3', 'bone_10':'L_Foot',
              'bone_11':'R_Foot', 'bone_12':'Neck', 'bone_13':'L_Collar', 'bone_14':'R_Collar',
              'bone_15':'Head', 'bone_16':'L_Shoulder', 'bone_17':'R_Shoulder', 'bone_18':'L_Elbow',
              'bone_19':'R_Elbow', 'bone_20':'L_Wrist', 'bone_21':'R_Wrist', 'bone_22':'L_Hand', 'bone_23':'R_Hand'}

part2num = {part:(ipart+1) for ipart,part in enumerate(sorted_parts)}

# create one material per part as defined in a pickle with the segmentation
# this is useful to render the segmentation in a material pass
def create_segmentation(ob, params):
    materials = {}
    vgroups = {}
    with open('pkl/segm_per_v_overlap.pkl', 'rb') as f:
        vsegm = load(f)
    bpy.ops.object.material_slot_remove()
    parts = sorted(vsegm.keys())
    for part in parts:
        vs = vsegm[part]
        vgroups[part] = ob.vertex_groups.new(part)
        vgroups[part].add(vs, 1.0, 'ADD')
        bpy.ops.object.vertex_group_set_active(group=part)
        materials[part] = bpy.data.materials['Material'].copy()
        materials[part].pass_index = part2num[part]
        bpy.ops.object.material_slot_add()
        ob.material_slots[-1].material = materials[part]
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.vertex_group_select()
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode='OBJECT')
    return(materials)

# create the different passes that we render
def create_composite_nodes(tree, params, idx=0):
    res_paths = {k:join(params['tmp_path'], '%05d_%s'%(idx, k)) for k in params['output_types'] if params['output_types'][k]}
    
    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create node for foreground image
    layers = tree.nodes.new('CompositorNodeRLayers')
    layers.location = -300, 400
    
    ## TODO : change this for an automatic generation of background scene
    # create node for background image
    
    if(params['output_types']['vblur']):
    # create node for computing vector blur (approximate motion blur)
        vblur = tree.nodes.new('CompositorNodeVecBlur')
        vblur.factor = params['vblur_factor']
        vblur.location = 240, 400

        # create node for saving output of vector blurred image 
        vblur_out = tree.nodes.new('CompositorNodeOutputFile')
        vblur_out.format.file_format = 'PNG'
        vblur_out.base_path = res_paths['vblur']
        vblur_out.location = 460, 460

    # create node for the final output 
    composite_out = tree.nodes.new('CompositorNodeComposite')
    composite_out.location = 240, 30
    
    #despeckle node
    despeckle = tree.nodes.new('CompositorNodeDespeckle')
    despeckle.threshold = 0.3

    # create node for saving depth
    if(params['output_types']['depth']):
        depth_out = tree.nodes.new('CompositorNodeOutputFile')
        depth_out.location = 40, 700
        depth_out.format.file_format = 'OPEN_EXR'
        depth_out.base_path = res_paths['depth']

    # create node for saving normals
    if(params['output_types']['normal']):
        normal_out = tree.nodes.new('CompositorNodeOutputFile')
        normal_out.location = 40, 600
        normal_out.format.file_format = 'OPEN_EXR'
        normal_out.base_path = res_paths['normal']

    # create node for saving foreground image
    if(params['output_types']['fg']):
        fg_out = tree.nodes.new('CompositorNodeOutputFile')
        fg_out.location = 170, 600
        fg_out.format.file_format = 'PNG'
        fg_out.base_path = res_paths['fg']

    # create node for saving ground truth flow 
    if(params['output_types']['gtflow']):
        gtflow_out = tree.nodes.new('CompositorNodeOutputFile')
        gtflow_out.location = 40, 500
        gtflow_out.format.file_format = 'OPEN_EXR'
        gtflow_out.base_path = res_paths['gtflow']

    # create node for saving segmentation
    if(params['output_types']['segm']):
        segm_out = tree.nodes.new('CompositorNodeOutputFile')
        segm_out.location = 40, 400
        segm_out.format.file_format = 'OPEN_EXR'
        segm_out.base_path = res_paths['segm']
    
        
        
    if(params['output_types']['vblur']):
        tree.links.new(layers.outputs['Image'], vblur.inputs[0])                # apply vector blur on the fg image,
        tree.links.new(layers.outputs['Depth'], vblur.inputs[1])           #   using depth, #replace ['Depth'] with ['Z'] for blender <2.79
        tree.links.new(layers.outputs['Vector'], vblur.inputs[2])       #   and flow. #replace ['Vector'] with ['Speed'] for blender <2.79
        tree.links.new(vblur.outputs[0], vblur_out.inputs[0])          # save vblurred output
         
    tree.links.new(layers.outputs[0], despeckle.inputs[1])    # save fg + bg
    tree.links.new(despeckle.outputs[0],composite_out.inputs[0])
    
    if(params['output_types']['fg']):
        tree.links.new(layers.outputs['Image'], fg_out.inputs[0])      # save fg
    if(params['output_types']['depth']):    
        tree.links.new(layers.outputs['Depth'], depth_out.inputs[0])       # save depth #replace ['Depth'] with ['Z'] for blender <2.79
    if(params['output_types']['normal']):
        tree.links.new(layers.outputs['Normal'], normal_out.inputs[0]) # save normal
    if(params['output_types']['gtflow']):
        tree.links.new(layers.outputs['Vector'], gtflow_out.inputs[0])  # save ground truth flow #replace ['Vector'] with ['Speed'] for blender <2.79
    if(params['output_types']['segm']):
        tree.links.new(layers.outputs['IndexMA'], segm_out.inputs[0])  # save segmentation

    return(res_paths)

# creation of the spherical harmonics material, using an OSL script
def create_sh_material(tree, sh_path, img=None):    

    for n in tree.nodes:
        tree.nodes.remove(n)
        
    uv = tree.nodes.new('ShaderNodeTexCoord')
    uv.location = -800, 400

    uv_xform = tree.nodes.new('ShaderNodeVectorMath')
    uv_xform.location = -600, 400
    uv_xform.inputs[1].default_value = (0, 0, 1)
    uv_xform.operation = 'AVERAGE'
    
    bsdf = tree.nodes.new("ShaderNodeBsdfDiffuse")
    mat_out = tree.nodes.new("ShaderNodeOutputMaterial")
    
    uv_im = tree.nodes.new('ShaderNodeTexImage')
    uv_im.location = -400, 400
    if img is not None:
        uv_im.image = img

    rgb = tree.nodes.new('ShaderNodeRGB')
    rgb.location = -400, 200
    
    tree.links.new(uv.outputs[2], uv_im.inputs[0])    
    tree.links.new(uv_im.outputs[0],bsdf.inputs[0])
    tree.links.new(bsdf.outputs[0], mat_out.inputs[0])

# computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

def init_scene(scene, params, gender='female'):
    
    # TODO : IMPORT A RANDOMLY CHOSEN 3D SCENE
    bpy.ops.object.select_all(action='DESELECT')  
    room_type = choice(['bedroom','living'])
    light_pos,light_size,light_intensity = build_background(room_type, [], width=20, length=15, height=2.4)
    
    # load fbx model
    bpy.ops.import_scene.fbx(filepath=join(params['smpl_data_folder'], 'basicModel_%s_lbs_10_207_0_v1.0.2.fbx' % gender[0]),
                             axis_forward='Y', axis_up='Z', global_scale=100)
    obname = '%s_avg' % gender[0] 
    ob = bpy.data.objects[obname]
    ob.data.use_auto_smooth = False  # autosmooth creates artifacts

    # assign the existing spherical harmonics material
    ob.active_material = bpy.data.materials['Material']

    # delete the default cube (which held the material)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete(use_global=False)
    
    
    # set camera properties and initial position
    bpy.ops.object.select_all(action='DESELECT')
    cam_ob = bpy.data.objects['Camera']
    scn = bpy.context.scene
    scn.objects.active = cam_ob

    cam_ob.matrix_world = Matrix(((0., 0., 1, params['camera_distance']),
                                 (0., -1, 0., -1.0),                                 
                                 (-1., 0., 0., 0.),
                                 (0.0, 0.0, 0.0, 1.0)))
    cam_ob.data.angle = math.radians(40)
    cam_ob.data.lens =  60
    cam_ob.data.clip_start = 0.1
    cam_ob.data.sensor_width = 32

    # setup an empty object in the center which will be the parent of the Camera
    # this allows to easily rotate an object around the origin
    scn.cycles.film_transparent = True
    scn.render.layers["RenderLayer"].use_pass_vector = True
    scn.render.layers["RenderLayer"].use_pass_normal = True
    scene.render.layers['RenderLayer'].use_pass_emit  = True
    scene.render.layers['RenderLayer'].use_pass_emit  = True
    scene.render.layers['RenderLayer'].use_pass_material_index  = True

    # set render size
    scn.render.resolution_x = params['resy']
    scn.render.resolution_y = params['resx']
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = 'PNG'

    # clear existing animation data
    ob.data.shape_keys.animation_data_clear()
    arm_ob = bpy.data.objects['Armature']
    arm_ob.animation_data_clear()

    return(ob, obname, arm_ob, cam_ob, light_pos, light_size, light_intensity)

# transformation between pose and blendshapes
def rodrigues2bshapes(pose):
    rod_rots = np.asarray(pose).reshape(24, 3)
    mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]
    bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel()
                              for mat_rot in mat_rots[1:]])
    return(mat_rots, bshapes)


# apply trans pose and shape to character
def apply_trans_pose_shape(trans, pose, shape, ob, arm_ob, obname, scene, cam_ob, frame=None):
    # transform pose into rotation matrices (for pose) and pose blendshapes
    mrots, bsh = rodrigues2bshapes(pose)

    # set the location of the first bone to the translation parameter
    arm_ob.pose.bones[obname+'_Pelvis'].location = trans
    if frame is not None:
        arm_ob.pose.bones[obname+'_root'].keyframe_insert('location', frame=frame)
    # set the pose of each bone to the quaternion specified by pose
    for ibone, mrot in enumerate(mrots):
        bone = arm_ob.pose.bones[obname+'_'+part_match['bone_%02d' % ibone]]
        bone.rotation_quaternion = Matrix(mrot).to_quaternion()
        if frame is not None:
            bone.keyframe_insert('rotation_quaternion', frame=frame)
            bone.keyframe_insert('location', frame=frame)

    # apply pose blendshapes
    for ibshape, bshape in enumerate(bsh):
        ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].value = bshape
        if frame is not None:
            ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)

    # apply shape blendshapes
    for ibshape, shape_elem in enumerate(shape):
        ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].value = shape_elem
        if frame is not None:
            ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)

def get_bone_locs(obname, arm_ob, scene, cam_ob):
    n_bones = 24
    render_scale = scene.render.resolution_percentage / 100
    render_size = (int(scene.render.resolution_x * render_scale),
                   int(scene.render.resolution_y * render_scale))
    bone_locations_2d = np.empty((n_bones, 2))
    bone_locations_3d = np.empty((n_bones, 3), dtype='float32')

    # obtain the coordinates of each bone head in image space
    for ibone in range(n_bones):
        bone = arm_ob.pose.bones[obname+'_'+part_match['bone_%02d' % ibone]]
        co_2d = world2cam(scene, cam_ob, arm_ob.matrix_world * bone.head)
        co_3d = arm_ob.matrix_world * bone.head
        bone_locations_3d[ibone] = (co_3d.x,
                                 co_3d.y,
                                 co_3d.z)
        bone_locations_2d[ibone] = (round(co_2d.x * render_size[0]),
                                 round(co_2d.y * render_size[1]))
    return(bone_locations_2d, bone_locations_3d)


# reset the joint positions of the character according to its new shape
def reset_joint_positions(orig_trans, shape, ob, arm_ob, obname, scene, cam_ob, reg_ivs, joint_reg):
    # since the regression is sparse, only the relevant vertex
    #     elements (joint_reg) and their indices (reg_ivs) are loaded
    reg_vs = np.empty((len(reg_ivs), 3))  # empty array to hold vertices to regress from
    # zero the pose and trans to obtain joint positions in zero pose
    apply_trans_pose_shape(orig_trans, np.zeros(72), shape, ob, arm_ob, obname, scene, cam_ob)

    # obtain a mesh after applying modifiers
    bpy.ops.wm.memory_statistics()
    # me holds the vertices after applying the shape blendshapes
    me = ob.to_mesh(scene, True, 'PREVIEW')

    # fill the regressor vertices matrix
    for iiv, iv in enumerate(reg_ivs):
        reg_vs[iiv] = me.vertices[iv].co
    bpy.data.meshes.remove(me)

    # regress joint positions in rest pose
    joint_xyz = joint_reg.dot(reg_vs)
    # adapt joint positions in rest pose
    arm_ob.hide = False
    bpy.ops.object.mode_set(mode='EDIT')
    arm_ob.hide = True
    for ibone in range(24):
        bb = arm_ob.data.edit_bones[obname+'_'+part_match['bone_%02d' % ibone]]
        bboffset = bb.tail - bb.head
        bb.head = joint_xyz[ibone]
        bb.tail = bb.head + bboffset
    bpy.ops.object.mode_set(mode='OBJECT')
    return(shape)

# load poses and shapes
def load_body_data(smpl_data, ob, obname, gender='female', idx=0):
    # load MoSHed data from CMU Mocap (only the given idx is loaded)
    
    # create a dictionary with key the sequence name and values the pose and trans
    cmu_keys = []
    for seq in smpl_data.files:
        if seq.startswith('pose_'):
            cmu_keys.append(seq.replace('pose_', ''))
    
    name = sorted(cmu_keys)[idx % len(cmu_keys)]
    
    cmu_parms = {}
    for seq in smpl_data.files:
        if seq == ('pose_' + name):
            cmu_parms[seq.replace('pose_', '')] = {'poses':smpl_data[seq],
                                                   'trans':smpl_data[seq.replace('pose_','trans_')]}

    # compute the number of shape blendshapes in the model
    n_sh_bshapes = len([k for k in ob.data.shape_keys.key_blocks.keys()
                        if k.startswith('Shape')])

    # load all SMPL shapes
    fshapes = smpl_data['%sshapes' % gender][:, :n_sh_bshapes]

    return(cmu_parms, fshapes, name)

import time
start_time = None
def log_message(message):
    elapsed_time = time.time() - start_time
    print("[%.2f s] %s" % (elapsed_time, message))

def main():
    # time logging
    global start_time
    start_time = time.time()

    import argparse
    
    # parse commandline arguments
    log_message(sys.argv)
    parser = argparse.ArgumentParser(description='Generate synth dataset images.')
    parser.add_argument('--idx', type=int,
                        help='idx of the requested sequence')
    parser.add_argument('--ishape', type=int,
                        help='requested cut, according to the stride')
    parser.add_argument('--stride', type=int,
                        help='stride amount, default 50')

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    
    idx = args.idx
    ishape = args.ishape
    stride = args.stride
    
    log_message("input idx: %d" % idx)
    log_message("input ishape: %d" % ishape)
    log_message("input stride: %d" % stride)
    
    if idx == None:
        exit(1)
    if ishape == None:
        exit(1)
    if stride == None:
        log_message("WARNING: stride not specified, using default value 50")
        stride = 50
    
    # import idx info (name, split)
    idx_info = load(open("pkl/idx_info.pickle", 'rb'))

    # get runpass
    (runpass, idx) = divmod(idx, len(idx_info))
    
    log_message("runpass: %d" % runpass)
    log_message("output idx: %d" % idx)
    idx_info = idx_info[idx]
    log_message("sequence: %s" % idx_info['name'])
    log_message("nb_frames: %f" % idx_info['nb_frames'])
    log_message("use_split: %s" % idx_info['use_split'])

    # import configuration
    log_message("Importing configuration")
    import config
    params = config.load_file('config', 'SYNTH_DATA')
    
    smpl_data_folder = params['smpl_data_folder']
    smpl_data_filename = params['smpl_data_filename']
    resy = params['resy']
    resx = params['resx']
    clothing_option = params['clothing_option'] # grey, nongrey or all
    tmp_path = params['tmp_path']
    output_path = params['output_path']
    output_types = params['output_types']
    stepsize = params['stepsize']
    clipsize = params['clipsize']
    openexr_py2_path = params['openexr_py2_path']

    # compute number of cuts
    nb_ishape = max(1, int(np.ceil((idx_info['nb_frames'] - (clipsize - stride))/stride)))
    log_message("Max ishape: %d" % (nb_ishape - 1))
    
    if ishape == None:
        exit(1)
    
    assert(ishape < nb_ishape)
    
    # name is set given idx
    name = idx_info['name']
    output_path = join(output_path, 'run%d' % runpass, name.replace(" ", ""))
    params['output_path'] = output_path
    tmp_path = join(tmp_path, 'run%d_%s_c%04d' % (runpass, name.replace(" ", ""), (ishape + 1)))
    params['tmp_path'] = tmp_path
    
    # check if already computed
    #  + clean up existing tmp folders if any
    if exists(tmp_path) and tmp_path != "" and tmp_path != "/":
        os.system('rm -rf %s' % tmp_path)
    rgb_vid_filename = "%s_c%04d.mp4" % (join(output_path, name.replace(' ', '')), (ishape + 1))
    #if os.path.isfile(rgb_vid_filename):
    #    log_message("ALREADY COMPUTED - existing: %s" % rgb_vid_filename)
    #    return 0
    
    # create tmp directory
    if not exists(tmp_path):
        mkdir_safe(tmp_path)
    
    # >> don't use random generator before this point <<

    # initialize RNG with seeds from sequence id
    import hashlib
    s = "synth_data:%d:%d:%d" % (idx, runpass,ishape)
    seed_number = int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
    log_message("GENERATED SEED %d from string '%s'" % (seed_number, s))
    random.seed(seed_number)
    np.random.seed(seed_number)
    
    if(output_types['vblur']):
        vblur_factor = np.random.normal(0.5, 0.5)
        params['vblur_factor'] = vblur_factor
    
    log_message("Setup Blender")

    # create copy-spher.harm. directory if not exists
    sh_dir = join(tmp_path, 'spher_harm')
    if not exists(sh_dir):
        mkdir_safe(sh_dir)
    sh_dst = join(sh_dir, 'sh_%02d_%05d.osl' % (runpass, idx))
    os.system('cp spher_harm/sh.osl %s' % sh_dst)

    genders = {0: 'female', 1: 'male'}
    # pick random gender
    gender = choice(genders)

    scene = bpy.data.scenes['Scene']
    scene.render.engine = 'CYCLES'    

    bpy.data.materials['Material'].use_nodes = True
    scene.use_nodes = True
  
    # grab clothing names
    log_message("clothing: %s" % clothing_option)
    with open( join(smpl_data_folder, 'textures', '%s_%s.txt' % ( gender, idx_info['use_split'] ) ) ) as f:
        txt_paths = f.read().splitlines()

    # if using only one source of clothing
    if clothing_option == 'nongrey':
        txt_paths = [k for k in txt_paths if 'nongrey' in k]
    elif clothing_option == 'grey':
        txt_paths = [k for k in txt_paths if 'nongrey' not in k]
    
    # random clothing texture
    cloth_img_name = choice(txt_paths)
    cloth_img_name = join(smpl_data_folder, cloth_img_name)
    cloth_img = bpy.data.images.load(cloth_img_name)
    
    log_message("Loading parts segmentation")
    beta_stds = np.load(join(smpl_data_folder, ('%s_beta_stds.npy' % gender)))
    
    log_message("Building materials tree")
    mat_tree = bpy.data.materials['Material'].node_tree
    create_sh_material(mat_tree, sh_dst, cloth_img)
    res_paths = create_composite_nodes(scene.node_tree, params, idx=idx)

    log_message("Loading smpl data")
    smpl_data = np.load(join(smpl_data_folder, smpl_data_filename))
    
    log_message("Initializing scene")
    camera_distance = np.random.normal(8.0, 1)
    params['camera_distance'] = camera_distance
    ob, obname, arm_ob, cam_ob, light_pos, light_size, light_intensity = init_scene(scene, params, gender)    

    setState0()
    ob.select = True
    bpy.context.scene.objects.active = ob
    segmented_materials = True #True: 0-24, False: expected to have 0-1 bg/fg
    
    log_message("Creating materials segmentation")
    # create material segmentation
    if segmented_materials:
        materials = create_segmentation(ob, params)
        prob_dressed = {'leftLeg':.5, 'leftArm':.9, 'leftHandIndex1':.01,
                        'rightShoulder':.8, 'rightHand':.01, 'neck':.01,
                        'rightToeBase':.9, 'leftShoulder':.8, 'leftToeBase':.9,
                        'rightForeArm':.5, 'leftHand':.01, 'spine':.9,
                        'leftFoot':.9, 'leftUpLeg':.9, 'rightUpLeg':.9,
                        'rightFoot':.9, 'head':.01, 'leftForeArm':.5,
                        'rightArm':.5, 'spine1':.9, 'hips':.9,
                        'rightHandIndex1':.01, 'spine2':.9, 'rightLeg':.5}
    else:
        materials = {'FullBody': bpy.data.materials['Material']}
        prob_dressed = {'FullBody': .6}

    orig_pelvis_loc = (arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname+'_Pelvis'].head.copy()) - Vector((-1., 1., 1.))
    orig_cam_loc = cam_ob.location.copy()

    # unblocking both the pose and the blendshape limits
    for k in ob.data.shape_keys.key_blocks.keys():
        bpy.data.shape_keys["Key"].key_blocks[k].slider_min = -10
        bpy.data.shape_keys["Key"].key_blocks[k].slider_max = 10

    log_message("Loading body data")
    cmu_parms, fshapes, name = load_body_data(smpl_data, ob, obname, idx=idx, gender=gender)
    
    log_message("Loaded body data for %s" % name)
    
    nb_fshapes = len(fshapes)
    if idx_info['use_split'] == 'train':
        fshapes = fshapes[:int(nb_fshapes*0.8)]
    elif idx_info['use_split'] == 'test':
        fshapes = fshapes[int(nb_fshapes*0.8):]
    
    # pick random real body shape
    shape = choice(fshapes) #+random_shape(.5) can add noise
    #shape = random_shape(3.) # random body shape
    
    # example shapes
    #shape = np.zeros(10) #average
    #shape = np.array([ 2.25176191, -3.7883464 ,  0.46747496,  3.89178988,  2.20098416,  0.26102114, -3.07428093,  0.55708514, -3.94442258, -2.88552087]) #fat
    #shape = np.array([-2.26781107,  0.88158132, -0.93788176, -0.23480508,  1.17088298,  1.55550789,  0.44383225,  0.37688275, -0.27983086,  1.77102953]) #thin
    #shape = np.array([ 0.00404852,  0.8084637 ,  0.32332591, -1.33163664,  1.05008727,  1.60955275,  0.22372946, -0.10738459,  0.89456312, -1.22231216]) #short
    #shape = np.array([ 3.63453289,  1.20836171,  3.15674431, -0.78646793, -1.93847355, -0.32129994, -0.97771656,  0.94531640,  0.52825811, -0.99324327]) #tall

    ndofs = 10

    scene.objects.active = arm_ob
    orig_trans = np.asarray(arm_ob.pose.bones[obname+'_Pelvis'].location).copy()

    # create output directory
    if not exists(output_path):
        mkdir_safe(output_path)
   
    rgb_dirname = name.replace(" ", "") + '_c%04d.mp4' % (ishape + 1)
    rgb_path = join(tmp_path, rgb_dirname)

    data = cmu_parms[name]
    
    fbegin = ishape*stepsize*stride
    fend = min(ishape*stepsize*stride + stepsize*clipsize, len(data['poses']))
    
    log_message("Computing how many frames to allocate")
    N = len(data['poses'][fbegin:fend:stepsize])
    log_message("Allocating %d frames in mat file" % N)

    # force recomputation of joint angles unless shape is all zeros
    curr_shape = np.zeros_like(shape)
    nframes = len(data['poses'][::stepsize])

    matfile_info = join(output_path, name.replace(" ", "") + "_c%04d_info.mat" % (ishape+1))
    log_message('Working on %s' % matfile_info)

    # allocate
    dict_info = {}
    dict_info['bg'] = np.zeros((N,), dtype=np.object) # background image path
    dict_info['camLoc'] = np.empty(3) # (1, 3)
    dict_info['clipNo'] = ishape +1
    dict_info['cloth'] = np.zeros((N,), dtype=np.object) # clothing texture image path
    dict_info['gender'] = np.empty(N, dtype='uint8') # 0 for male, 1 for female
    dict_info['joints2D'] = np.empty((2, 24, N), dtype='float32') # 2D joint positions in pixel space
    dict_info['joints3D'] = np.empty((3, 24, N), dtype='float32') # 3D joint positions in world coordinates
    dict_info['light'] = np.empty((5, N), dtype='float32')
    dict_info['pose'] = np.empty((data['poses'][0].size, N), dtype='float32') # joint angles from SMPL (CMU)
    dict_info['sequence'] = name.replace(" ", "") + "_c%04d" % (ishape + 1)
    dict_info['shape'] = np.empty((ndofs, N), dtype='float32')
    dict_info['zrot'] = np.empty(N, dtype='float32')
    dict_info['camDist'] = camera_distance
    dict_info['stride'] = stride

    if name.replace(" ", "").startswith('h36m'):
        dict_info['source'] = 'h36m'
    else:
        dict_info['source'] = 'cmu'

    if(output_types['vblur']):
        dict_info['vblur_factor'] = np.empty(N, dtype='float32')

    # for each clipsize'th frame in the sequence
    get_real_frame = lambda ifr: ifr
    random_zrot = 0
    reset_loc = False
    batch_it = 0
    curr_shape = reset_joint_positions(orig_trans, shape, ob, arm_ob, obname, scene,
                                       cam_ob, smpl_data['regression_verts'], smpl_data['joint_regressor'])
    random_zrot = 2*np.pi*np.random.rand()
    
    arm_ob.animation_data_clear()
    cam_ob.animation_data_clear()

    # create a keyframe animation with pose, translation, blendshapes and camera motion
    # LOOP TO CREATE 3D ANIMATION
    for seq_frame, (pose, trans) in enumerate(zip(data['poses'][fbegin:fend:stepsize], data['trans'][fbegin:fend:stepsize])):
        iframe = seq_frame
        scene.frame_set(get_real_frame(seq_frame))

        # apply the translation, pose and shape to the character
        apply_trans_pose_shape(Vector(trans), pose, shape, ob, arm_ob, obname, scene, cam_ob, get_real_frame(seq_frame))
        dict_info['shape'][:, iframe] = shape[:ndofs]
        dict_info['pose'][:, iframe] = pose
        dict_info['gender'][iframe] = list(genders)[list(genders.values()).index(gender)]
        if(output_types['vblur']):
            dict_info['vblur_factor'][iframe] = vblur_factor

        arm_ob.pose.bones[obname+'_root'].rotation_quaternion = Quaternion(Euler((0, 0, random_zrot), 'XYZ'))
        arm_ob.pose.bones[obname+'_root'].keyframe_insert('rotation_quaternion', frame=get_real_frame(seq_frame))
        dict_info['zrot'][iframe] = random_zrot

        scene.update()

        # Bodies centered only in each minibatch of clipsize frames
        if seq_frame == 0 or reset_loc: 
            reset_loc = False
            new_pelvis_loc = arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname+'_Pelvis'].head.copy()
            cam_ob.location = orig_cam_loc.copy() + (new_pelvis_loc.copy() - orig_pelvis_loc.copy())
            cam_ob.keyframe_insert('location', frame=get_real_frame(seq_frame))
            dict_info['camLoc'] = np.array(cam_ob.location)

    for part, material in materials.items():
        material.node_tree.nodes['Vector Math'].inputs[1].default_value[:2] = (0, 0)    
    
    #set GPU    
    sysp = bpy.context.user_preferences.system
#    devt = sysp.compute_device_type = 'CUDA'
#    dev = sysp.compute_device = 'CUDA_0'
    scene.cycles.device = 'GPU'   
    bpy.context.scene.render.layers[0].cycles.use_denoising = True # only for blender 2.79
    scene.cycles.samples = 128 
    bpy.ops.object.select_all(action='DESELECT') 
    
#    scene.cycles.sample_clamp_direct = 6.0
#    scene.cycles.sample_clamp_indirect = 3.0
    
    
    for obj in bpy.data.objects:
        if obj.name=='Sphere' or obj.name=='Armature'  or obj.name=='f_avg':
            obj.cycles_visibility.diffuse = True #This divides the rendering time by two
            obj.cycles_visibility.glossy = True        
        else:
            obj.cycles_visibility.diffuse = False
            obj.cycles_visibility.glossy = True
    
#    for obj in bpy.data.objects:
#        obj.cycles_visibility.diffuse = True #This divides the rendering time by two
#        obj.cycles_visibility.glossy = True        
#        if obj.name=='Room' or obj.name=='Floor':
#            obj.cycles_visibility.diffuse = False
#            obj.cycles_visibility.glossy = True
        
    
    
    # iterate over the keyframes and render
    # LOOP TO RENDER
    for seq_frame, (pose, trans) in enumerate(zip(data['poses'][fbegin:fend:stepsize], data['trans'][fbegin:fend:stepsize])):
        scene.frame_set(get_real_frame(seq_frame))
        iframe = seq_frame

        dict_info['bg'][iframe] = 'auto_bg'
        dict_info['cloth'][iframe] = cloth_img_name
        li = [light_pos, light_size, light_intensity]
        dict_info['light'][:, iframe] = [i for sub in li for i in sub]

        	
        
        
        scene.render.use_antialiasing = False
        scene.render.filepath = join(rgb_path, 'Image%04d.png' % get_real_frame(seq_frame))

        log_message("Rendering frame %d" % seq_frame)
        
        # disable render output
        logfile = '/dev/null'
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)

        # Render
        bpy.ops.render.render(write_still=True)

        # disable output redirection
        os.close(1)
        os.dup(old)
        os.close(old)

        # NOTE:
        # ideally, pixels should be readable from a viewer node, but I get only zeros
        # --> https://ammous88.wordpress.com/2015/01/16/blender-access-render-results-pixels-directly-from-python-2/
        # len(np.asarray(bpy.data.images['Render Result'].pixels) is 0
        # Therefore we write them to temporary files and read with OpenEXR library (available for python2) in main_part2.py
        # Alternatively, if you don't want to use OpenEXR library, the following commented code does loading with Blender functions, but it can cause memory leak.
        # If you want to use it, copy necessary lines from main_part2.py such as definitions of dict_normal, matfile_normal...

        #for k, folder in res_paths.items():
        #   if not k== 'vblur' and not k=='fg':
        #       path = join(folder, 'Image%04d.exr' % get_real_frame(seq_frame))
        #       render_img = bpy.data.images.load(path)
        #       # render_img.pixels size is width * height * 4 (rgba)
        #       arr = np.array(render_img.pixels[:]).reshape(resx, resy, 4)[::-1,:, :] # images are vertically flipped 
        #       if k == 'normal':# 3 channels, original order
        #           mat = arr[:,:, :3]
        #           dict_normal['normal_%d' % (iframe + 1)] = mat.astype(np.float32, copy=False)
        #       elif k == 'gtflow':
        #           mat = arr[:,:, 1:3]
        #           dict_gtflow['gtflow_%d' % (iframe + 1)] = mat.astype(np.float32, copy=False)
        #       elif k == 'depth':
        #           mat = arr[:,:, 0]
        #           dict_depth['depth_%d' % (iframe + 1)] = mat.astype(np.float32, copy=False)
        #       elif k == 'segm':
        #           mat = arr[:,:,0]
        #           dict_segm['segm_%d' % (iframe + 1)] = mat.astype(np.uint8, copy=False)
        #
        #       # remove the image to release memory, object handles, etc.
        #       render_img.user_clear()
        #       bpy.data.images.remove(render_img)

        # bone locations should be saved after rendering so that the bones are updated
        bone_locs_2D, bone_locs_3D = get_bone_locs(obname, arm_ob, scene, cam_ob)
        dict_info['joints2D'][:, :, iframe] = np.transpose(bone_locs_2D)
        dict_info['joints3D'][:, :, iframe] = np.transpose(bone_locs_3D)

        reset_loc = (bone_locs_2D.max(axis=-1) > 256).any() or (bone_locs_2D.min(axis=0) < 0).any()
        arm_ob.pose.bones[obname+'_root'].rotation_quaternion = Quaternion((1, 0, 0, 0))

    # save a .blend file for debugging:
    # bpy.ops.wm.save_as_mainfile(filepath=join(tmp_path, 'pre.blend'))
    
    # save RGB data with ffmpeg (if you don't have h264 codec, you can replace with another one and control the quality with something like -q:v 3)
    cmd_ffmpeg = 'ffmpeg -y -r 30 -i ''%s'' -c:v h264 -pix_fmt yuv420p -crf 23 ''%s_c%04d.mp4''' % (join(rgb_path, 'Image%04d.png'), join(output_path, name.replace(' ', '')), (ishape + 1))
    log_message("Generating RGB video (%s)" % cmd_ffmpeg)
    os.system(cmd_ffmpeg)
    
    if(output_types['vblur']):
        cmd_ffmpeg_vblur = 'ffmpeg -y -r 30 -i ''%s'' -c:v h264 -pix_fmt yuv420p -crf 23 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ''%s_c%04d.mp4''' % (join(res_paths['vblur'], 'Image%04d.png'), join(output_path, name.replace(' ', '')+'_vblur'), (ishape + 1))
        log_message("Generating vblur video (%s)" % cmd_ffmpeg_vblur)
        os.system(cmd_ffmpeg_vblur)
   
    if(output_types['fg']):
        cmd_ffmpeg_fg = 'ffmpeg -y -r 30 -i ''%s'' -c:v h264 -pix_fmt yuv420p -crf 23 ''%s_c%04d.mp4''' % (join(res_paths['fg'], 'Image%04d.png'), join(output_path, name.replace(' ', '')+'_fg'), (ishape + 1))
        log_message("Generating fg video (%s)" % cmd_ffmpeg_fg)
        os.system(cmd_ffmpeg_fg)
   
    cmd_tar = 'tar -czvf %s/%s.tar.gz -C %s %s' % (output_path, rgb_dirname, tmp_path, rgb_dirname)
    log_message("Tarballing the images (%s)" % cmd_tar)
    os.system(cmd_tar)
    
    # save annotation excluding png/exr data to _info.mat file
    import scipy.io
    scipy.io.savemat(matfile_info, dict_info, do_compression=True)

if __name__ == '__main__':
    main()
