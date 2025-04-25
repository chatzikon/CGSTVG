from torch import nn
from torch.nn.functional import dropout

from .net_utils import MLP
from .vision_model import build_vis_encoder
from .language_model import build_text_encoder
from .grounding_model import build_encoder, build_decoder
from utils.misc import NestedTensor
from .vidswin.video_swin_transformer import vidswin_model

from new.vjepa import build_vjepa_encoder, build_vjepa_classifier, vjepa_predict, VJEPAConfig
import torch

from models.vision_model.position_encoding import build_position_encoding

from  models.grounding_model.position_encoding import SeqEmbeddingLearned, SeqEmbeddingSine

from models.bert_model.bert_module import BertLayerNorm


def modality_concatenation(self,feat_2d, feat_motion, feat_text,pos_motion,pos_rgb):



    # expand the attention mask and text token from [b, len] to [n_frames, len]
    frame_length = feat_2d.size()[0]
    text_features = feat_text.expand(feat_text.size(0), frame_length,feat_text.size(-1))  # [text_len, n_frames, d_model]


        # concat visual and text features and Pad the vis_pos with 0 for the text tokens
    concat_features = torch.cat([feat_2d.permute(1,0,2), text_features, feat_motion.permute(1,0,2)], dim=0)
    #vis_pos = torch.cat([pos_motion, torch.zeros_like(text_features), pos_rgb], dim=0)

    frames_cls = torch.mean(concat_features, dim=0)
    videos_cls = torch.mean(frames_cls, dim=0)

    pos_query, content_query = self.pos_fc(frames_cls), self.time_fc(videos_cls)

    pos_query = pos_query.sigmoid()  # [n_frames, bs, 4]
    content_query = content_query.expand(self.NCLIPS*self.FRAMES_PER_CLIP, content_query.size(-1)).unsqueeze(1)  # [n_frames, bs, d_model]

    conf_query=self.conf(pos_query).sigmoid().squeeze()

    return pos_query, content_query, conf_query




class CGSTVG(nn.Module):
    def __init__(self, cfg):
        super(CGSTVG, self).__init__()
        self.cfg = cfg.clone()
        self.max_video_len = cfg.INPUT.MAX_VIDEO_LEN
        self.use_attn = cfg.SOLVER.USE_ATTN

        self.device='cuda:0'
        
        self.use_aux_loss = cfg.SOLVER.USE_AUX_LOSS  # use the output of each transformer layer
        self.use_actioness = cfg.MODEL.CG.USE_ACTION
        self.query_dim = cfg.MODEL.CG.QUERY_DIM

        #self.vis_encoder = build_vis_encoder(cfg)
        #vis_fea_dim = self.vis_encoder.num_channels
      
        self.text_encoder = build_text_encoder(cfg)
        
        #self.ground_encoder = build_encoder(cfg)
        #self.ground_decoder = build_decoder(cfg)
        
        hidden_dim = cfg.MODEL.CG.HIDDEN
        #self.input_proj = nn.Conv2d(vis_fea_dim, hidden_dim, kernel_size=1)
        self.temp_embed = MLP(hidden_dim, hidden_dim, 2, 2, dropout=0.3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        #self.vid = vidswin_model("video_swin_t_p4w7", "video_swin_t_p4w7_k400_1k")
        #self.input_proj2 = nn.Conv2d(768, hidden_dim, kernel_size=1)
        #for param in self.vid.parameters():
        #    param.requires_grad = False

        self.action_embed = None
        if self.use_actioness:
            self.action_embed = MLP(hidden_dim, hidden_dim, 1, 2, dropout=0.3)

        #self.ground_decoder.time_embed2 = self.action_embed

        # add the iteration anchor update
        #self.ground_decoder.decoder.bbox_embed = self.bbox_embed
        
        #### V-JEPA extension ####
        self.vjepa_config = VJEPAConfig()
        self.vjepa_encoder = build_vjepa_encoder(self.vjepa_config)

        self.vjepa_classifier_motion = build_vjepa_classifier(
            config=self.vjepa_config,
            encoder=self.vjepa_encoder,
            video_data=True,
            checkpoint_path="JEPA/app/model_zoo/k400-probe.pth.tar"
        )

        self.vjepa_classifier_2d = build_vjepa_classifier(
            config=self.vjepa_config,
            encoder=self.vjepa_encoder,
            video_data=False,
            checkpoint_path="JEPA/app/model_zoo/in1k-probe.pth.tar"
        )

        self.NCLIPS = 8
        self.VIEWS_PER_CLIP = 1
        self.FRAMES_PER_CLIP = 16
        self.B = 1



        ###embeds
        self.motion_embed = MLP(1,(self.NCLIPS*self.FRAMES_PER_CLIP)//2, self.NCLIPS*self.FRAMES_PER_CLIP,2, dropout=0.3)
        self.rgb_embed = MLP(1, (self.NCLIPS*self.FRAMES_PER_CLIP)//2, self.NCLIPS*self.FRAMES_PER_CLIP,2, dropout=0.3)


        self.encoder_motion_embed= nn.Linear((hidden_dim*hidden_dim)//(self.NCLIPS*self.FRAMES_PER_CLIP), self.vjepa_config.num_classes_vid, bias=True)
        self.encoder_rgb_embed = nn.Linear((hidden_dim*hidden_dim)//(self.NCLIPS*self.FRAMES_PER_CLIP), self.vjepa_config.num_classes_img, bias=True)

        self.mask_motion_embed=nn.Linear(self.vjepa_config.num_classes_vid, hidden_dim, bias=True)
        self.mask_rgb_embed=nn.Linear(self.vjepa_config.num_classes_img,hidden_dim, bias=True)

        ###Decoders####
        decoder_layer_2d = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=cfg.MODEL.CG.HEADS)
        self.decoder_2d = nn.TransformerDecoder(decoder_layer_2d, num_layers=cfg.MODEL.CG.DEC_LAYERS//3)

        decoder_layer_motion = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=cfg.MODEL.CG.HEADS)
        self.decoder_motion = nn.TransformerDecoder(decoder_layer_motion, num_layers=cfg.MODEL.CG.DEC_LAYERS//3)

        decoder_layer_text = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=cfg.MODEL.CG.HEADS)
        self.decoder_text = nn.TransformerDecoder(decoder_layer_text, num_layers=cfg.MODEL.CG.DEC_LAYERS//3)

        self.pos_fc = nn.Sequential(
            BertLayerNorm(256, eps=1e-12),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
            nn.ReLU(True),
            BertLayerNorm(4, eps=1e-12),
        )

        self.time_fc = nn.Sequential(
            BertLayerNorm(256, eps=1e-12),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(True),
            BertLayerNorm(256, eps=1e-12),
        )

        self.conf = MLP(4, self.NCLIPS*self.FRAMES_PER_CLIP, 1, 3,  dropout=0.3)

        return



    def forward(self, videos, texts, targets, iteration_rate=-1):



        T, C, H, W = videos.tensors.shape
        clips = videos.tensors.reshape(shape=(self.NCLIPS, self.VIEWS_PER_CLIP, self.B, C, self.FRAMES_PER_CLIP, H, W))
        clip_indices = torch.reshape(targets[0]["frame_ids"], (self.NCLIPS, self.FRAMES_PER_CLIP))
        

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=self.vjepa_config.use_bfloat16):
            with torch.no_grad():
                vjepa_features = self.vjepa_encoder(clips, clip_indices)

            if self.vjepa_config.attend_across_segments:
                outputs_motion = [self.vjepa_classifier_motion(o) for o in vjepa_features]
                outputs_2d = [self.vjepa_classifier_2d(o) for o in vjepa_features]

            else:
                outputs_motion = [[self.vjepa_classifier_motion(ost) for ost in os] for os in vjepa_features]
                outputs_2d = [[self.vjepa_classifier_2d(ost) for ost in os] for os in vjepa_features]


            mask_motion=self.motion_embed(torch.permute(outputs_motion[0],(1,0)))
            mask_rgb = self.rgb_embed(torch.permute(outputs_2d[0],(1,0)))

            d_model = self.cfg.MODEL.CG.HIDDEN
            tgt = torch.zeros(self.NCLIPS*self.FRAMES_PER_CLIP, self.B, d_model).to(self.device)


            ####positional embedding backbone
            position_embedding = build_position_encoding(self.cfg)


            jepa_features=torch.permute(vjepa_features[0],(1,2,0))
            mask_temp=torch.zeros((jepa_features.size()[0],jepa_features.size()[2]), dtype=torch.bool).to(self.device)
            # False
            encoder_pos=position_embedding(torch.unsqueeze(jepa_features,-1),torch.unsqueeze(mask_temp,-1))

            encoder_pos_sq=torch.squeeze(encoder_pos)
            encoder_pos=torch.permute(encoder_pos_sq.view(-1, self.NCLIPS*self.FRAMES_PER_CLIP),(1,0))

            encoder_pos_motion=self.encoder_motion_embed(encoder_pos)
            encoder_pos_rgb = self.encoder_rgb_embed(encoder_pos)

            ####positional embedding encoder
            if self.cfg.MODEL.CG.USE_LEARN_TIME_EMBED:
                self.tgt_embed = SeqEmbeddingLearned(self.NCLIPS*self.FRAMES_PER_CLIP + 1, d_model)
            else:
                self.tgt_embed = SeqEmbeddingSine(self.NCLIPS*self.FRAMES_PER_CLIP + 1, d_model)
            tgt_pos_motion = self.tgt_embed(self.NCLIPS*self.FRAMES_PER_CLIP).to(self.device)
            tgt_pos_rgb = self.tgt_embed(self.NCLIPS*self.FRAMES_PER_CLIP).to(self.device)



            mask_motion_1=torch.unsqueeze(torch.permute(mask_motion,(1,0)),1)
            mask_rgb_1 = torch.unsqueeze(torch.permute(mask_rgb, (1, 0)), 1)
            encoder_pos_motion_1 = torch.unsqueeze(encoder_pos_motion, 1)
            encoder_pos_rgb_1 = torch.unsqueeze(encoder_pos_rgb, 1)


            mask_motion_f=self.mask_motion_embed(mask_motion_1)
            mask_rgb_f=self.mask_rgb_embed(mask_rgb_1)
            encoder_pos_motion_f = self.mask_motion_embed(encoder_pos_motion_1)
            encoder_pos_rgb_f = self.mask_rgb_embed(encoder_pos_rgb_1)







                ###visual decoding
            output_motion=self.decoder_motion(tgt+tgt_pos_motion,mask_motion_f+encoder_pos_motion_f)
            output_2d = self.decoder_2d(tgt+tgt_pos_rgb, mask_rgb_f+encoder_pos_rgb_f)



            # # Visual Feature
            # vis_outputs, vis_pos_embed = self.vis_encoder(videos)
            # vis_features, vis_mask, vis_durations = vis_outputs.decompose()
            # vis_features = self.input_proj(vis_features)
            # vis_outputs = NestedTensor(vis_features, vis_mask, vis_durations)
            #
            # vid_features = self.vid(videos.tensors, len(videos.tensors))
            # vid_features = self.input_proj2(vid_features['3']) # [128, 256, 1, 2]

            # Textual Feature
            device = clips.device
            text_outputs, _ = self.text_encoder(texts, device)

            #position_embedding = build_position_encoding(self.cfg)
            #vis_pos = position_embedding(feat_2d)


            pos_query, time_query, conf_query =modality_concatenation(self,output_2d, output_motion, text_outputs[1],encoder_pos_motion_f,encoder_pos_rgb_f)
            ####Decoding

            # Multimodal Feature Encoding
            # encoded_info = self.ground_encoder(videos=vis_outputs, vis_pos=vis_pos_embed, texts=text_outputs, vid_features=vid_features)
            # encoded_info["iteration_rate"] = iteration_rate
            # encoded_info["videos"] = videos
            # # Query-based Decoding
            # outputs_pos, outputs_time = self.ground_decoder(encoded_info=encoded_info, vis_pos=vis_pos_embed, targets=targets)

            out = {}

            # the final decoder embeddings and the refer anchors
            ###############  predict bounding box ###############

            #outputs_coord = refer_anchors.flatten(1,2)  # [num_layers, T, 4]
            out.update({"pred_boxes": pos_query})
            out.update({"boxes_conf": conf_query})
            ######################################################

            #######  predict the start and end probability #######
            time_hiden_state = time_query
            outputs_time = self.temp_embed(time_hiden_state)  # [num_layers, b, T, 2]
            out.update({"pred_sted": torch.permute(outputs_time,(1,0,2))})
            #######################################################

            if self.use_actioness:
                outputs_actioness = self.action_embed(time_hiden_state)  # [num_layers, b, T, 1]
                out.update({"pred_actioness": outputs_actioness[-1]})

            if self.use_aux_loss:
                out["aux_outputs"] = [
                    {
                        "pred_sted": a,
                        "pred_boxes": b,
                        "boxes_conf": c
                    }
                    for a, b, c in zip(outputs_time[:-1], pos_query, conf_query)
                ]
                for i_aux in range(len(out["aux_outputs"])):
                    if self.use_actioness:
                        out["aux_outputs"][i_aux]["pred_actioness"] = outputs_actioness[i_aux]

        return out

