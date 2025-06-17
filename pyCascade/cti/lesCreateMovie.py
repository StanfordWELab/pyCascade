# modified from /home/groups/gorle/codes/fidelityCharles_2024.1/fidelityCharles_2024.2_CTI/scripts/lesCreateMovie
import sys
sys.path.append('/home/groups/gorle/codes/fidelityCharles_2024.1/fidelityCharles_2024.2_CTI/scripts')
sys.path.append('/home/groups/gorle/codes/fidelityCharles_2024.1/fidelityCharles_2024.2_CTI/scripts/cti')
sys.path.append('/home/groups/gorle/codes/fidelityCharles_2024.1/fidelityCharles_2024.2_CTI/scripts/pythonModules')


import cti_image
import  scrOpts as so
import  scrTools as st


import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib import colormaps
import numpy as np
import glob

#---------------------------------------------------------------------------
# func definitions
#---------------------------------------------------------------------------

def create_cbar(data_min, data_max, cm, nticks, title, cb_image_name, cb_w, cb_h, cbar_orient, 
                background_color=[73, 175, 205], fontsize=14):
    """Create a colorbar image
    
    Args:
        data_min: Minimum value for the colorbar
        data_max: Maximum value for the colorbar
        cm: Colormap to use
        nticks: Number of ticks on the colorbar
        title: Title for the colorbar
        cb_image_name: Output image filename
        cb_w: Width of the colorbar in inches
        cb_h: Height of the colorbar in inches
        cbar_orient: Orientation of the colorbar ('horizontal' or 'vertical')
        background_color: RGB background color (default: bahama blue)
        fontsize: Font size for labels (default: 14)
    
    Returns:
        Path to the created colorbar image
    """
    fig, ax = plt.subplots(figsize=(cb_w, cb_h))
    fig.subplots_adjust(bottom=0.3)

    fig.patch.set_facecolor((background_color[0]/255, background_color[1]/255, background_color[2]/255))
    ticks = np.linspace(data_min, data_max, nticks)
    if data_max - data_min > 1:
        ticks = np.round(ticks).astype(int)
    else:
        ticks = np.round(ticks, decimals=3)
    
    norm = plt.Normalize(vmin=np.min(ticks), vmax=np.max(ticks))
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm),
                        cax=ax, orientation=cbar_orient)
    cbar.set_label(title, fontsize=fontsize)
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=fontsize)

    cbar_path = cb_image_name
    plt.savefig(cbar_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    return cbar_path

def cbar_padding(cb_loc, img_h, img_w, cb_images, nvar, cbar_orient, background_color=[73, 175, 205]):
    """Pad colorbar images to match dimensions
    
    Args:
        cb_loc: Location of the colorbar
        img_h: Height of the image
        img_w: Width of the image
        cb_images: List of colorbar images
        nvar: Number of variables
        cbar_orient: Orientation of colorbars
        background_color: RGB background color (default: bahama blue)
    
    Returns:
        List of padded colorbar images
    """
    cb_h_max = max(array.shape[0] for array in cb_images)
    cb_w_max = max(array.shape[1] for array in cb_images)
    
    padded_cb = []
    if nvar==1:
        cb_h = cb_images[0].shape[0]
        cb_w = cb_images[0].shape[1]
        if cbar_orient=='horizontal':
            p_left = (img_w - cb_w)//2; p_right = img_w - cb_w - p_left
            p_top = 0; p_bottom = 0
        elif cbar_orient=='vertical':
            p_left = 0; p_right = 0
            p_top = (img_h - cb_h)//2; p_bottom = img_h - cb_h - p_top
        p_cb = cv.copyMakeBorder(
            cb_images[0],
            top=int(p_top), bottom=int(p_bottom), 
            left=int(p_left), right=int(p_right),
            borderType=cv.BORDER_CONSTANT, value=(background_color[2],background_color[1],background_color[0])
        )
        padded_cb.append(p_cb)
        
    elif nvar==2:
        cb_h = cb_images[0].shape[0]
        cb_w = cb_images[0].shape[1] 
        if cbar_orient=='horizontal':
            # left aligned
            p_left = 0; p_right = img_w - cb_w - p_left
            p_top = 0; p_bottom = cb_h_max - cb_h
        elif cbar_orient=='vertical':
            # top aligned
            p_left = 0; p_right = cb_w_max - cb_w
            p_top = 0; p_bottom = img_h - cb_h - p_top
        p_cb_1 = cv.copyMakeBorder(
            cb_images[0],
            top=int(p_top), bottom=int(p_bottom), 
            left=int(p_left), right=int(p_right),
            borderType=cv.BORDER_CONSTANT, value=(background_color[2],background_color[1],background_color[0])
        )
        
        cb_h = cb_images[1].shape[0]
        cb_w = cb_images[1].shape[1] 
        if cbar_orient=='horizontal':
            # right aligned
            p_right = 0; p_left = img_w - cb_w - p_right; 
            p_top = 0; p_bottom = cb_h_max - cb_h
        elif cbar_orient=='vertical':
            # bottom aligned
            p_left = 0; p_right = cb_w_max - cb_w
            p_bottom = 0; p_top = img_h - cb_h - p_bottom; 
        p_cb_2 = cv.copyMakeBorder(
            cb_images[1],
            top=int(p_top), bottom=int(p_bottom), 
            left=int(p_left), right=int(p_right),
            borderType=cv.BORDER_CONSTANT, value=(background_color[2],background_color[1],background_color[0])
        )

        p_cb = (p_cb_1 + p_cb_2) 
        p_cb[:,:,0] = p_cb[:,:,0] - background_color[2]
        p_cb[:,:,1] = p_cb[:,:,1] - background_color[1]
        p_cb[:,:,2] = p_cb[:,:,2] - background_color[0]
        padded_cb.append(p_cb)

    elif nvar==3:
        cb_h = cb_images[0].shape[0]
        cb_w = cb_images[0].shape[1] 
        if cbar_orient=='horizontal':
            # left aligned
            p_left = 0; p_right = img_w - cb_w - p_left
            p_top = 0; p_bottom = cb_h_max - cb_h
        elif cbar_orient=='vertical':
            # top aligned
            p_left = 0; p_right = cb_w_max - cb_w
            p_top = 0; p_bottom = img_h - cb_h - p_top
        p_cb_1 = cv.copyMakeBorder(
            cb_images[0],
            top=int(p_top), bottom=int(p_bottom), 
            left=int(p_left), right=int(p_right),
            borderType=cv.BORDER_CONSTANT, value=(background_color[2],background_color[1],background_color[0])
        )
        
        cb_h = cb_images[1].shape[0]
        cb_w = cb_images[1].shape[1] 
        if cbar_orient=='horizontal':
            # right aligned
            p_right = 0; p_left = img_w - cb_w - p_right; 
            p_top = 0; p_bottom = cb_h_max - cb_h
        elif cbar_orient=='vertical':
            # bottom aligned
            p_left = 0; p_right = cb_w_max - cb_w
            p_bottom = 0; p_top = img_h - cb_h - p_bottom; 
        p_cb_2 = cv.copyMakeBorder(
            cb_images[1],
            top=int(p_top), bottom=int(p_bottom), 
            left=int(p_left), right=int(p_right),
            borderType=cv.BORDER_CONSTANT, value=(background_color[2],background_color[1],background_color[0])
        )

        p_cb = (p_cb_1 + p_cb_2)
        p_cb[:,:,0] = p_cb[:,:,0] - background_color[2]
        p_cb[:,:,1] = p_cb[:,:,1] - background_color[1]
        p_cb[:,:,2] = p_cb[:,:,2] - background_color[0]
        padded_cb.append(p_cb)

        cb_h = cb_images[2].shape[0]
        cb_w = cb_images[2].shape[1]
        if cbar_orient=='horizontal':
            # left aligned top alone
            p_left = 0; p_right = img_w - cb_w - p_left
            p_top = 0; p_bottom = 0
        elif cbar_orient=='vertical':
            # top aligned right alone
            p_left = 0; p_right = 0
            p_top = 0; p_bottom = img_h - cb_h - p_top

        p_cb_3 = cv.copyMakeBorder(
            cb_images[2],
            top=int(p_top), bottom=int(p_bottom), 
            left=int(p_left), right=int(p_right),
            borderType=cv.BORDER_CONSTANT, value=(background_color[2],background_color[1],background_color[0])
        )
        padded_cb.append(p_cb_3)

    elif nvar==4:
        cb_h = cb_images[0].shape[0]
        cb_w = cb_images[0].shape[1] 
        if cbar_orient=='horizontal':
            # left aligned
            p_left = 0; p_right = img_w - cb_w - p_left
            p_top = 0; p_bottom = cb_h_max - cb_h
        elif cbar_orient=='vertical':
            # top aligned
            p_left = 0; p_right = cb_w_max - cb_w
            p_top = 0; p_bottom = img_h - cb_h - p_top
        p_cb_1 = cv.copyMakeBorder(
            cb_images[0],
            top=int(p_top), bottom=int(p_bottom), 
            left=int(p_left), right=int(p_right),
            borderType=cv.BORDER_CONSTANT, value=(background_color[2],background_color[1],background_color[0])
        )
        
        cb_h = cb_images[1].shape[0]
        cb_w = cb_images[1].shape[1] 
        if cbar_orient=='horizontal':
            # right aligned
            p_right = 0; p_left = img_w - cb_w - p_right; 
            p_top = 0; p_bottom = cb_h_max - cb_h
        elif cbar_orient=='vertical':
            # bottom aligned
            p_left = 0; p_right = cb_w_max - cb_w
            p_bottom = 0; p_top = img_h - cb_h - p_bottom; 
        p_cb_2 = cv.copyMakeBorder(
            cb_images[1],
            top=int(p_top), bottom=int(p_bottom), 
            left=int(p_left), right=int(p_right),
            borderType=cv.BORDER_CONSTANT, value=(background_color[2],background_color[1],background_color[0])
        )

        p_cb = (p_cb_1 + p_cb_2)
        p_cb[:,:,0] = p_cb[:,:,0] - background_color[2]
        p_cb[:,:,1] = p_cb[:,:,1] - background_color[1]
        p_cb[:,:,2] = p_cb[:,:,2] - background_color[0]
        padded_cb.append(p_cb)

        cb_h = cb_images[2].shape[0]
        cb_w = cb_images[2].shape[1] 
        if cbar_orient=='horizontal':
            # left aligned top 
            p_left = 0; p_right = img_w - cb_w - p_left
            p_top = cb_h_max - cb_h; p_bottom = 0
        elif cbar_orient=='vertical':
            # top aligned 
            p_left = 0; p_right = cb_w_max - cb_w
            p_top = 0; p_bottom = img_h - cb_h - p_top
        p_cb_3 = cv.copyMakeBorder(
            cb_images[2],
            top=int(p_top), bottom=int(p_bottom), 
            left=int(p_left), right=int(p_right),
            borderType=cv.BORDER_CONSTANT, value=(background_color[2],background_color[1],background_color[0])
        )

        cb_h = cb_images[3].shape[0]
        cb_w = cb_images[3].shape[1]
        if cbar_orient=='horizontal':
            # right aligned top 
            p_right = 0; p_left = img_w - cb_w - p_right 
            p_top = cb_h_max - cb_h; p_bottom = 0
        elif cbar_orient=='vertical':
            # bottom aligned 
            p_left = 0; p_right = cb_w_max - cb_w
            p_bottom = 0; p_top = img_h - cb_h - p_bottom
        p_cb_4 = cv.copyMakeBorder(
            cb_images[3],
            top=int(p_top), bottom=int(p_bottom), 
            left=int(p_left), right=int(p_right),
            borderType=cv.BORDER_CONSTANT, value=(background_color[2],background_color[1],background_color[0])
        )
        p_cb = (p_cb_3 + p_cb_4)
        p_cb[:,:,0] = p_cb[:,:,0] - background_color[2]
        p_cb[:,:,1] = p_cb[:,:,1] - background_color[1]
        p_cb[:,:,2] = p_cb[:,:,2] - background_color[0]
        padded_cb.append(p_cb)

    return padded_cb

def process_image(image_path, varlist, cmaplist=None, data_min=None, data_max=None):

    im = cti_image.Image(image_path)
    im.getImageMetadataAndChunks()
    img = im.getRGB()
    for v, var in enumerate(varlist):
        new_image = img.copy()
        if im.daTa is not None and cmaplist is not None: # and im.flAg is not None:
            color_mapped_img = plt.get_cmap(cmaplist[v])(im.chunks['daTa']/255.0)
        else:
            img_gs = np.mean(new_image, axis=-1) / 255
            if data_min is not None and data_max is not None: 
                var_min = im.metadata[var]['range'][0]
                var_max = im.metadata[var]['range'][1]
                img_gs = img_gs * (var_max - var_min) + var_min # scale up with local range
                img_gs = (img_gs - data_min[v]) / (data_max[v] - data_min[v]) # scale down with global range

            if cmaplist is not None:
                color_mapped_img = plt.get_cmap(cmaplist[v])(img_gs)
            else:
                color_mapped_img = np.stack((img_gs, img_gs, img_gs), axis=-1)

        color_mapped_img = (color_mapped_img[:,:,:3] * 255).astype(np.uint8)

        if im.flAg is not None:
            mask = im.getMask(var)
        else:
            mask = im.zoNe > 100

        new_image[mask] = color_mapped_img[mask]
        color_mapped_img = cv.cvtColor(new_image.astype(np.uint8), cv.COLOR_RGB2BGR)


    return color_mapped_img


def get_varTypes(image_path, params):
    """Get variable types and their properties from the image
    
    Args:
        image_path: Path to the image file
        params: Dictionary containing configuration parameters
        
    Returns:
        varslist: List of variable types
        cmaplist: List of colormaps for each variable
        cb_names: List of colorbar image names
        data_min: List of minimum data values
        data_max: List of maximum data values
        titles: List of colorbar titles
    """
    varslist = ['planar']
    cmaplist = [params['colormap_planar']]
    cb_names = [params['prefix'] + 'colorbar.png']
    titles = [params['cbar_title']]

    im = cti_image.Image(image_path)
    im.getImageMetadataAndChunks()
    data_min = [im.metadata['planar']['range'][0]]
    data_max = [im.metadata['planar']['range'][1]]

    if im.metadata['iso']['available'] == True:
        varslist.append('iso')
        cmaplist.append(params['colormap_iso'])
        cb_names.append(params['prefix'] + 'colorbar_iso.png')
        data_min.append(im.metadata['iso']['range'][0])
        data_max.append(im.metadata['iso']['range'][1])
        titles.append(params['cbar_iso_title'])

    if im.metadata['surf']['available'] == True:
        varslist.append('surf')
        cmaplist.append(params['colormap_surf'])
        cb_names.append(params['prefix'] + 'colorbar_surf.png')
        data_min.append(im.metadata['surf']['range'][0])
        data_max.append(im.metadata['surf']['range'][1])
        titles.append(params['cbar_surf_title'])

    if im.metadata['particle']['available'] == True:
        varslist.append('particle')
        cmaplist.append(params['colormap_particle'])
        cb_names.append(params['prefix'] + 'colorbar_particle.png')
        data_min.append(im.metadata['particle']['range'][0])
        data_max.append(im.metadata['particle']['range'][1])
        titles.append(params['cbar_particle_title'])

    return varslist, cmaplist, cb_names, data_min, data_max, titles
    

def validate_parameters(params):
    """Check the parameters for validity"""
    if params['infile'] == '':
        st.scrErr("No input file given")

    params['output'] = params['prefix'] + params['movie_filename'] + '.mp4'
    params['colormap_planar'] = params['colormap']

    all_mpl_cmaps = list(colormaps)
    if params['colormap_planar'] not in all_mpl_cmaps:
        st.scrErr("colormap planar: " + params['colormap_planar'] + " not available in matplotlib")
    if params['colormap_iso'] not in all_mpl_cmaps:
        st.scrErr("colormap iso: " + params['colormap_iso'] + " not available in matplotlib")
    if params['colormap_surf'] not in all_mpl_cmaps:
        st.scrErr("colormap surf: " + params['colormap_surf'] + " not available in matplotlib")
    if params['colormap_particle'] not in all_mpl_cmaps:
        st.scrErr("colormap particle: " + params['colormap_particle'] + " not available in matplotlib")
    
    return params

def print_parameters(params):
    """Print the parameters for user information"""
    st.scrPrint('input params:')
    st.scrPrint('  infile  = %s' % params['infile'])
    st.scrPrint('  prefix  = %s' % params['prefix'])
    st.scrPrint('  movie_filename  = %s' % params['movie_filename'])
    st.scrPrint('  fps  = %s' % params['fps'])
    st.scrPrint('  add_cbar_movie  = %s' % params['add_cbar_movie'])
    st.scrPrint('  colormap  = %s' % params['colormap'])
    st.scrPrint('  colormap_iso  = %s' % params['colormap_iso'])
    st.scrPrint('  colormap_surf  = %s' % params['colormap_surf'])
    st.scrPrint('  colormap_particle  = %s' % params['colormap_particle'])
    st.scrPrint('  cbar_title  = %s' % params['cbar_title'])
    st.scrPrint('  cbar_iso_title  = %s' % params['cbar_iso_title'])
    st.scrPrint('  cbar_surf_title  = %s' % params['cbar_surf_title'])
    st.scrPrint('  cbar_particle_title  = %s' % params['cbar_particle_title'])
    st.scrPrint('  cbar_width_frac  = %s' % params['cbar_width_frac'])
    st.scrPrint('  cbar_height_frac  = %s' % params['cbar_height_frac'])
    st.scrPrint('  cbar_orient  = %s' % params['cbar_orient'])
    st.scrPrint('  nticks  = %s' % params['nticks'])
    st.scrPrint('  fontsize  = %s' % params['fontsize'])

def create_video(params):
    """Main function to create the video"""
    # Check and print params
    video_base = f'{params["image_base"]}{params["video_suffix"]}'
    params["infile"] = f'{params["image_folder"]}/{params["image_base"]}*.png'
    params["movie_filename"] = f'{params["video_folder"]}/{video_base}'
    params = validate_parameters(params)
    print_parameters(params)

    # Define parameters that were previously global
    background_color = [255, 255, 255] #[73, 175, 205]  # bahama blue rgb
    fontsize = params['fontsize']
    dpi = 85
    
    # Get image files
    files = sorted(glob.glob(params['infile']))
    
    # Get variable types and necessary information
    vars_list, cmaplist, cb_names, data_min, data_max, titles = get_varTypes(files[-1], params)
    nvar = len(vars_list)
    
    # Process the last image to get dimensions
    last_image = process_image(files[-1], vars_list, cmaplist, data_min, data_max)
    image_width = last_image.shape[1]
    image_height = last_image.shape[0]
    cbar_width = params['cbar_width_frac'] * image_width / dpi
    cbar_height = params['cbar_height_frac'] * image_height / dpi
    
    # Create colorbar images
    cbar_img = []
    for v, var in enumerate(vars_list):
        cbar_path = create_cbar(data_min[v], data_max[v], cmaplist[v], params['nticks'], 
                               titles[v], cb_names[v], cbar_width, cbar_height, 
                               params['cbar_orient'], background_color, fontsize)
        cbar_img.append(cv.imread(cbar_path))
    
    # Prepare final image dimensions
    if params['add_cbar_movie']:
        cb_resized = cbar_padding(None, last_image.shape[0], last_image.shape[1], 
                                cbar_img, nvar, params['cbar_orient'], background_color)
        
        if params['cbar_orient'] == 'horizontal':
            if len(cb_resized) == 1:
                last_image_combined = np.vstack((last_image, cb_resized[0]))
            elif len(cb_resized) == 2:
                last_image_combined = np.vstack((cb_resized[1], last_image, cb_resized[0]))
        elif params['cbar_orient'] == 'vertical':
            if len(cb_resized) == 1:
                last_image_combined = np.hstack((last_image, cb_resized[0]))
            elif len(cb_resized) == 2:
                last_image_combined = np.hstack((cb_resized[1], last_image, cb_resized[0]))
                
        h, w, _ = last_image_combined.shape
    else:
        h, w, _ = last_image.shape
    
    # Set up video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    frame_size = (w, h)
    out = cv.VideoWriter(params['output'], fourcc, params['fps'], frame_size)
    
    # Process each frame and add to video
    for img_path in files:
        print(img_path)
        frame = process_image(img_path, vars_list, cmaplist, data_min, data_max)
        
        if params['add_cbar_movie']:
            if params['cbar_orient'] == 'horizontal':
                if len(cb_resized) == 1:
                    combined_frame = np.vstack((frame, cb_resized[0]))
                elif len(cb_resized) == 2:
                    combined_frame = np.vstack((cb_resized[1], frame, cb_resized[0]))
            elif params['cbar_orient'] == 'vertical':
                if len(cb_resized) == 1:
                    combined_frame = np.hstack((frame, cb_resized[0]))
                elif len(cb_resized) == 2:
                    combined_frame = np.hstack((cb_resized[1], frame, cb_resized[0]))
        else:
            combined_frame = frame
            
        out.write(combined_frame)
    
    out.release()
    print(f"Video saved to {params['output']}")

def get_default_params():
    params = {
        "image_base":"u_y1p5",
        "image_folder":"//scratch/users/nbachand/Cascade/city_block_cfd/CHARLES/config2/R53/Images",
        "video_folder":"/oak/stanford/groups/gorle/nbachand/Cascade/city_block_cfd/CHARLES/config2/R53/Videos",
        "video_suffix":"_pyVid",
        "prefix":"",
        "fps":30,
        "colormap":"plasma",
        "colormap_iso":"seismic",
        "colormap_surf":"coolwarm",
        "colormap_particle":"hot",
        "cbar_orient":"vertical",
        "cbar_width_frac":0.02,
        "cbar_height_frac":0.45,
        "fontsize":14,
        "add_cbar_movie":True,
        "nticks":5,
        "cbar_title":"planar_data",
        "cbar_iso_title":"iso_data",
        "cbar_surf_title":"surface_data",
        "cbar_particle_title":"particle data"
    }

    return params

def main():
    """Main function to orchestrate the video creation process"""
    # Set up options
    params = get_default_params()

    # Create the video
    create_video(params)

if __name__ == "__main__":
    main()