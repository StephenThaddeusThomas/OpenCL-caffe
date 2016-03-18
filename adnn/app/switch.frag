// ENABLE SWITCH when both RunSingleConvLayerInfer.o and RunSingleLayerTrain.o are both compiled and linked 
#ifdef ENABLE_SWITCH  
{
  if(training)
    return(run_single_conv_layer_training(aLib,
					  &layer_control,
					  &f_params,
					  batch_sz,
					  input_channels,
					  input_h,
					  input_w,
					  n_output_features,
					  &update_params));
  else
    return(run_single_conv_layer_infere(aLib,
					&layer_control,
					&f_params,
					batch_sz,
					input_channels,
					input_h,
					input_w,
					n_output_features,
					&update_params));
}		   
int run_single_conv_layer_infere(alib_obj aLib,
		    const adnn_control_params *layer_control,
		    const adnn_filter1D_parameters *filter_params,
		    int batch_sz,
		    int input_channels,
		    int input_h,
		    int input_w,
		    int n_output_featuremaps,
		    adnn_update_params *pupdate_params)
#endif
