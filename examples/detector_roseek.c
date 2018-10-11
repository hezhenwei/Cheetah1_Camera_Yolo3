#include "darknet.h"

#include "CTMainCore.h"

image get_roseek_image() {
    
    int c = 1;
    int i,j,k;
    int w = 1920;
    int h = 1080;

    image im = make_image(w, h, c);

	Roseek_ImageAcquisition_GrabSingleFrame();

    CT_RAWFRAME *pFrame;
    Roseek_Capture_FetchFconFrame(&pFrame, 100);
	if( NULL == pFrame) {
		return im;
	}
	char* pData = (char*)pFrame->pDataBuf;

    for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = (float)pData[src_index];
            }
        }
    }

    if (pFrame)
    {
        Roseek_Capture_ReleaseFrame(pFrame);
    }

    return im;
}

void test_detector_roseek(char *datacfg, char *cfgfile, char *weightfile, float thresh, float hier_thresh, int fullscreen)
{
    int ret = 0;
	ret = Roseek_MainCore_Init();
	if (ret != 0)
	{
		return;
	}

	ret = Roseek_Capture_Build();
	if (ret != 0)
	{
		return;
	}

	Roseek_CapturingParameters_SetExposureMode(CT_RUNMODE_TRG, CT_EXPOSUREMODE_MANUAL);
	Roseek_CapturingParameters_SetExposureTime(CT_RUNMODE_TRG, 20000);
	Roseek_CapturingParameters_SetGain(CT_RUNMODE_TRG, 0);
	Roseek_ImageAcquisition_SetRunMode(CT_RUNMODE_TRG);


    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
    while(1){
	// input is the file name...so need to modify to 
        image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net->w, net->h);
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n-1];


        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        //printf("%d\n", nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        free_detections(dets, nboxes);

        save_image(im, "predictions");
#ifdef OPENCV
        make_window("predictions", 512, 512, 0);
        show_image(im, "predictions", 0);
#endif

        free_image(im);
        free_image(sized);
    }

	Roseek_Capture_Destroy();
	Roseek_MainCore_UnInit();
}

