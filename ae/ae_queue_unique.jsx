/* Reviewed native queue setup only. Never renders or overwrites an AEP.
 * Fill CONFIG after the actual host exposes a project-owned EXR32 output
 * output template. Its readback must pass here; real file headers must then
 * prove FLOAT32 (an output-module Depth label alone does not prove EXR32).
 */
var CONFIG = __CONFIG_JSON__;
(function () {
    function need(ok, message) { if (!ok) { throw new Error(message); } }
    function escape(value) { return String(value).replace(/[\\"\x00-\x1f\u2028\u2029]/g, function (c) {
        return "\\u" + ("0000" + c.charCodeAt(0).toString(16)).slice(-4); }); }
    function json(value) {
        var type = typeof value, parts = [], key, i;
        if (value === null || type === "undefined") { return "null"; }
        if (type === "string") { return "\"" + escape(value) + "\""; }
        if (type === "number") { return isFinite(value) ? String(value) : "null"; }
        if (type === "boolean") { return value ? "true" : "false"; }
        if (value instanceof Array) { for (i=0; i<value.length; i+=1) { parts.push(json(value[i])); } return "["+parts.join(",")+"]"; }
        for (key in value) { if (Object.prototype.hasOwnProperty.call(value,key) && typeof value[key] !== "function") { parts.push("\""+escape(key)+"\":"+json(value[key])); } }
        return "{"+parts.join(",")+"}";
    }
    function normalized(path) { return String(path).replace(/\\/g,"/").toLowerCase(); }
    var result={ok:false,rendered:false,implementation:"4k-native100-queue-v3-span-start",phase:CONFIG.phase,items:[],expected_output_files:[]};
    var receipt=new File(CONFIG.receipt_path),undoOpen=false;
    need(!receipt.exists&&receipt.parent.exists,"NEW_RECEIPT_REQUIRED");
    function saveReceipt(){receipt.encoding="UTF-8";need(!receipt.exists&&receipt.open("w"),"NEW_RECEIPT_WRITE_FAILED");try{receipt.write(json(result)+"\n");}finally{receipt.close();}}
    try {
        var project=app.project;
        need(project!==null&&project.file!==null&&project.dirty===false,"SAVED_CLEAN_OWNED_PROJECT_REQUIRED");
        need(normalized(project.file.fsName)===normalized(new File(CONFIG.expected_current_project).fsName),"USER_PROJECT_PRESERVED_UNEXPECTED_CURRENT_PROJECT");
        if(CONFIG.open_base_project===true){project=app.open(new File(CONFIG.base_project));}
        need(project!==null&&project.file!==null&&normalized(project.file.fsName)===normalized(new File(CONFIG.base_project).fsName),"EXPECTED_BASE_PROJECT_REQUIRED");
        need(project.bitsPerChannel===32&&(project.workingSpace===""||project.workingSpace==="None")
             &&project.linearBlending===false&&project.linearizeWorkingSpace===false,"NATIVE_COLOR_CONTRACT_CHANGED");
        need(project.renderQueue.numItems===0,"EXISTING_QUEUE_PRESERVED");
        need(CONFIG.phase==="probe"||CONFIG.phase==="full","EXPLICIT_PHASE_REQUIRED");
        if(CONFIG.phase==="full"){need(CONFIG.probe_numeric_passed===true&&CONFIG.probe_A_error<=0.0000002384185791015625&&CONFIG.probe_RGB_error<=0.0000002384185791015625,"ACTUAL_ORIGINAL_F_A_INPUT_PROBE_REQUIRED");}
        var targetProject=new File(CONFIG.render_project_path);
        need(!targetProject.exists&&targetProject.parent.exists,"NEW_RENDER_PROJECT_REQUIRED");
        var specs=CONFIG.specs;
        var seen={},total=0;
        app.beginUndoGroup("Native4K EXR queue - no rendering");undoOpen=true;
        for(var s=0;s<specs.length;s+=1){
            var spec=specs[s],comp=null;
            for(var i=1;i<=project.numItems;i+=1){if(project.item(i) instanceof CompItem&&project.item(i).name===spec.comp){need(comp===null,"DUPLICATE_COMP_NAME");comp=project.item(i);}}
            need(comp!==null&&comp.width===4096&&comp.height===2160&&comp.frameRate===25&&Math.round(comp.duration*25)===65,"NATIVE4K_CLOCK_OR_COMP_MISMATCH");
            var target=new File(CONFIG.render_root+"/"+spec.folder+"/"+spec.prefix+"_[#####].exr");need(target.parent.exists,"OUTPUT_DIRECTORY_REQUIRED");
            need(!seen[normalized(target.parent.fsName)]&&target.parent.getFiles("*.exr").length===0,"UNIQUE_EMPTY_DIRECTORY_PER_QUEUE_ITEM_REQUIRED");
            seen[normalized(target.parent.fsName)]=true;total+=spec.count;
            var item=project.renderQueue.items.add(comp);item.applyTemplate("Best Settings");
            item.timeSpanStart=spec.first/25;item.timeSpanDuration=spec.count/25;
            var module=item.outputModule(1);module.applyTemplate(CONFIG.output_template);module.file=target;
            var settings=module.getSettings(GetSettingsFormat.STRING);
            need(String(settings["Format"]).toLowerCase().indexOf("openexr")>=0&&String(settings["Depth"]).toLowerCase().indexOf("float")>=0,"EXR_FLOAT_FORMAT_REQUIRED");
            need(String(settings["Channels"]).toLowerCase().indexOf("alpha")>=0&&String(settings["Color"]).toLowerCase().indexOf("premultiplied")>=0,"PREMULTIPLIED_RGBA_REQUIRED");
            need(String(settings["Resize"]).toLowerCase()==="false"&&String(settings["Use Region of Interest"]).toLowerCase()==="false","NO_RESIZE_OR_ROI_REQUIRED");
            need(parseInt(String(settings["Starting #"]),10)===spec.first&&String(settings["Use Comp Frame Number"]).toLowerCase()==="true","ORIGINAL_COMP_FRAME_NUMBER_REQUIRED");
            result.items.push({branch:spec.branch,output_directory:target.parent.fsName,comp:spec.comp,first_local_frame:spec.first,frame_count:spec.count,output_pattern:module.file.fsName,
                settings:settings,render_settings:item.getSettings(GetSettingsFormat.STRING)});
        }
        need(total===CONFIG.expected_exports,"EXACT_EXPORT_COUNT_REQUIRED");
        result.new_exports=total;result.output_template=CONFIG.output_template;
        var viewerComp=null;
        for(var c=1;c<=project.numItems;c+=1){if(project.item(c) instanceof CompItem&&project.item(c).name==="COMPOSITE_CLEAN_REFINED_4K"){viewerComp=project.item(c);}}
        need(viewerComp!==null,"FINAL_CLEAN_GRAY_COMP_REQUIRED");viewerComp.time=8/25;viewerComp.openInViewer();
        app.endUndoGroup();undoOpen=false;
        project.save(targetProject);need(targetProject.exists&&project.dirty===false,"RENDER_PROJECT_SAVE_FAILED");
        result.render_project_path=project.file.fsName;result.project_bits_per_channel=project.bitsPerChannel;
        result.working_space_raw=project.workingSpace;result.default_open_comp=viewerComp.name;result.default_open_local_frame=8;
        result.alpha_association="premultiplied/black";result.color_contract="Existing LTX_BF16_EXR32_CODE_ASSOC template; PreserveRGB/LinearOff behavior must pass native pixel checks, FLOAT32/ZIP must pass actual file headers.";
        result.ok=true;result.status="QUEUED_SAVED_NOT_RENDERED";saveReceipt();return json(result);
    }catch(error){result.error=String(error);result.status="FAILED_PARTIAL_STATE_PRESERVED";if(!receipt.exists){try{saveReceipt();}catch(ignore){}}throw error;}
    finally{if(undoOpen){app.endUndoGroup();}}
}());
