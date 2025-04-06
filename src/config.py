class Config:

    def __init__(self):
        
        # Apply translation and rotation to make it more difficult.
        self.scramble = True

        # Apply classic Reinhard stain normalization if using standard H&E pathology images.
        # YMMV for other stains. 
        self.do_stain_normalization = True

        # The downsampled size that is used for the general reconstruction pipeline. 
        self.optimal_image_size = 2000
        
        # The spacing in µm/px at which you want to save the final reconstructed images.
        self.full_resolution_spacing = 8
        
        # Minimum number of images required for a meaningful reconstruction
        self.min_images_for_reconstruction = 3

        # Some specimen sectioning variables. The slice thickness represents the thickness (micron) of one 
        # finalized slice of tissue on a glass slide. The slice distance represents the distance between two
        # adjacent slices in micron. In case one glass slide is prepared per tissue block, this is simply 
        # equal to the thickness of the tissue blocks. These values hold for the prostatectomies from our center
        # but may differ for your own data.
        self.slice_thickness = 4 
        self.slice_distance = 4000 

        return

