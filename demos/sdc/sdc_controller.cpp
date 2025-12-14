#include "sdc_controller.h"

std::pair<int, int> encode_unorm6(
    float x
) {
    assert(x >= 0.0f && x <= 1.0f);

    Byte b = static_cast<Byte>(x * 63.f + 0.5f);

    return std::make_pair(b & 0x07, (b & 0x38) >> 3);
}

float decode_unorm6(
    const std::pair<int, int> &x
) {
    assert(x.first >= 0 && x.first < 8 && x.second >= 0 && x.second < 8);

    return (x.first | (x.second << 3)) / 63.0f;
}

void SDC_Controller::init_random() {
    Array<Image_Encoder::Visible_Layer_Desc> evlds(1);

    evlds[0].size = scfg.image_size;
    evlds[0].radius = scfg.enc_radius;

    enc.init_random(scfg.enc_size, evlds);

    Array<Hierarchy::Layer_Desc> lds(scfg.layer_sizes.size());

    for (int l = 0; l < lds.size(); l++) {
        lds[l].hidden_size = scfg.layer_sizes[l];
        lds[l].up_radius = scfg.layer_up_radius;
        lds[l].down_radius = scfg.layer_down_radius;
        lds[l].num_dendrites_per_cell = scfg.layer_num_dendrites_per_cell;
    }

    Array<Hierarchy::IO_Desc> iods(2);

    iods[0].type = scfg.gen_enabled ? prediction : none;
    iods[0].size = scfg.enc_size;
    iods[0].up_radius = scfg.io_up_radius;
    iods[0].down_radius = scfg.io_down_radius;
    iods[0].num_dendrites_per_cell = scfg.io_num_dendrites_per_cell;

    iods[1].type = scfg.rl_enabled ? action : prediction;
    iods[1].size = Int3(2, 2, 8);
    iods[1].up_radius = 1;
    iods[1].down_radius = scfg.io_down_radius;
    iods[1].num_dendrites_per_cell = scfg.io_num_dendrites_per_cell;
    iods[1].value_num_dendrites_per_cell = scfg.io_value_num_dendrites_per_cell;

    h.init_random(iods, lds);
}

bool SDC_Controller::init_load(
    const std::string &file_name
) {
    File_Reader reader;
    reader.ins.open(file_name, std::ios::binary);

    if (!reader.ins.is_open()) {
        std::cerr << "Could not load SPH!" << std::endl;

        return false;
    }

    enc.read(reader);
    h.read(reader);

    return true;
}

void SDC_Controller::save(
    const std::string &file_name
) {
    File_Writer writer;
    writer.outs.open(file_name, std::ios::binary);

    enc.write(writer);
    h.write(writer);
}

std::pair<float, float> SDC_Controller::step(
    const std::vector<unsigned char> &image,
    SDC_Mode mode,
    float reward,
    const std::pair<float, float> &targets
) {
    assert(image.size() == scfg.image_size.x * scfg.image_size.y * scfg.image_size.z);

    // encode image
    Byte_Buffer image_buffer(scfg.image_size.x * scfg.image_size.y * scfg.image_size.z);

    for (int i = 0; i < image_buffer.size(); i++)
        image_buffer[i] = image[i];

    Array<Byte_Buffer_View> images(1);

    images[0] = image_buffer;

    enc.step(images, mode != SDC_Mode::inf, scfg.gen_enabled);

    Array<Int_Buffer_View> inputs(2);

    inputs[0] = enc.get_hidden_cis();

    Int_Buffer target_cis(4);
    inputs[1] = target_cis;

    switch (mode) {
    case SDC_Mode::bc: {
        std::pair<int, int> pt = encode_unorm6(targets.first * 0.5f + 0.5f);
        std::pair<int, int> ps = encode_unorm6(targets.second * 0.5f + 0.5f);

        target_cis[0] = pt.first;
        target_cis[1] = pt.second;
        target_cis[2] = ps.first;
        target_cis[3] = ps.second;

        h.step(inputs, true, reward, 1.0f);

        break;
    }

    case SDC_Mode::rl: {
        target_cis = h.get_prediction_cis(1);
	
        h.step(inputs, true, reward, 0.0f);

        break;
    }

    case SDC_Mode::inf: {
        target_cis = h.get_prediction_cis(1);

        h.step(inputs, false, reward, 0.0f);

        break;
    }
    };

    const Int_Buffer &pred = h.get_prediction_cis(1);

    float pred_throttle = decode_unorm6({ pred[0], pred[1] }) * 2.0f - 1.0f;
    float pred_steer = decode_unorm6({ pred[2], pred[3] }) * 2.0f - 1.0f;

    smooth_throttle += scfg.action_smoothing * (pred_throttle - smooth_throttle);
    smooth_steer += scfg.action_smoothing * (pred_steer - smooth_steer);

    // get result
    return std::make_pair(smooth_throttle, smooth_steer);
}
