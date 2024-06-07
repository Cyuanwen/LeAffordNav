'''
分析找不到物体的原因：
1. 探索策略无效，视野中没有出现物体
2. 感知模型无效，视野中出现了物体，识别不出来

找不到物体，分两类：
1. 找不到goal object: a) 找不到start recep; b)找不到 goal object
2. 找不到end recep
'''
from typing import Optional
import os
import json


class find_error_ana:
    def __init__(self,
        results_path:str,
        detect_error_path:str,
        log_path:Optional[str]=None
    ) -> None:
        '''
            results_path: 结果记录文件夹，如 datadump_debug/results/eval_hssd_cyw
            detect_error_path: 检测失败结果路径，如 gyzp/output/detect_error/yolo_only
        '''
        self.results_path = results_path
        self.detect_error_path = detect_error_path
        with open(os.path.join(self.results_path,"episode_results.json"),"r") as f:
            self.results = json.load(f)
        
    def fail_found(self, episode_result):
        '''分析当前episode是否没有找到物体
        Argument:
            episode_result: 
        Return:
            fial_found: 是否没有找到物体
            fail_found_object: 没找到的物体是什么
        '''
        goal_name = episode_result['goal_name'].split()
        goal_object = goal_name[1]
        start_recep = goal_name[3]
        end_recep = goal_name[5]
        if not episode_result["END.ovmm_find_object_phase_success"]:
            return True, [goal_object,start_recep]
        elif episode_result["END.ovmm_pick_object_phase_success"] and not episode_result["END.ovmm_find_recep_phase_success"]:
            return True, [end_recep]
        else:
            return False, None
    
    def object_appear(self, detect_results_path,goal_object,is_goal):
        '''分析物体是否出现过在机器人视野中
        Arguments:
            detect_results_path: 检测结果路径
            object:要分析的物体
            is_goal:该物体是否是目标小物体
        Return:
            whether the object appear in robot view
        '''
        has_appear = False
        if is_goal:
            if os.path.exists(os.path.join(detect_results_path,"goal","error_labels")):
                error_labels = os.listdir(os.path.join(detect_results_path,"goal","error_labels"))
                if len(error_labels)!= 0:
                    return True
        else:
            if os.path.exists(os.path.join(detect_results_path,"receptacle","error_labels")):
                error_labels = os.listdir(os.path.join(detect_results_path,"receptacle","error_labels"))
                for error_label in error_labels:
                    with open(os.path.join(detect_results_path,"receptacle","error_labels",error_label),"r") as f:
                        lines = f.readlines()
                    for line in lines:
                        if goal_object in line:
                            return True
        return has_appear
        

    def fail_found_error(self):
        '''分析没有发现物体的原因
        '''
        fail_dict = {}
        for episode_key, episode_result in self.results.items():
            error_info = ""
            fail_found,fail_objects = self.fail_found(episode_result)
            detect_results_path = os.path.join(self.detect_error_path,episode_key)
            if not os.path.exists(detect_results_path):
                print(f"the results {detect_results_path} is not exists")
                break
            if fail_found:
                if len(fail_objects)==2:
                    goal_obj_appear = self.object_appear(detect_results_path,fail_objects[0],True)
                    if not goal_obj_appear:
                        start_recp_appear = self.object_appear(detect_results_path,fail_objects[1],False)
                        if not start_recp_appear:
                            error_info = f"start recep and goal object didn't appear: {fail_objects[1]} and {fail_objects[0]}"
                        else:
                            error_info = f"goal object didn't appear: {fail_objects[0]}"
                    else:
                        error_info = f"goal object miss detection: {fail_objects[0]}"
                else:
                    end_recep_appear = self.object_appear(detect_results_path,fail_objects[0],False)
                    if not end_recep_appear:
                        error_info = f"end recep didn't appear: {fail_objects[0]}"
                    else:
                        error_info = f"end recep miss detection: {fail_objects[0]}"
                fail_dict[episode_key] = error_info
        # 记录结果
        with open(os.path.join(self.results_path,"find_error.json"),"w") as f:
            json.dump(fail_dict,f,indent=2)
        

if __name__ == "__main__":
    results_path = "datadump/results/eval_hssd_cyw_print_img_yolo_main_gather"
    detect_record_path = "datadump/detect_error/eval_hssd_cyw_print_img_yolo_main_gather"
    analyser = find_error_ana(
        results_path=results_path,
        detect_error_path=detect_record_path
    )
    analyser.fail_found_error()

                        






        
