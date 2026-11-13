/*
 * Reviewed-source template for vfx-pipeline ae-run (ExtendScript / ES3).
 * Replace the CONFIG initializer token with json.dumps(manifest,
 * ensure_ascii=True). Do not execute this unfilled template directly.
 * Build only after full65 input verification and later main host authorization.
 * No rendering, preview export, effects or color corrections are performed.
 */
var CONFIG = __CONFIG_JSON__;

(function () {
    var receipt = {
        ok: false,
        stage: "preflight",
        implementation: "ltx-bf16-4k-derived-refined-ML-editable-v2",
        project_saved: false,
        project_changed: false,
        visual_review_required: true
    };
    var receiptFile = null;
    var undoOpen = false;
    var project = null;
    var GEOMETRY_EPSILON = 0.0001;
    var MAX_FRAME_RESIDUAL = 0.001;

    function requireCondition(condition, code) {
        if (!condition) {
            throw new Error(code);
        }
    }

    function escapeString(value) {
        return String(value).replace(/[\\"\x00-\x1f\u2028\u2029]/g, function (character) {
            var hex = character.charCodeAt(0).toString(16);
            return "\\u" + ("0000" + hex).slice(-4);
        });
    }

    function serialize(value) {
        var kind = typeof value;
        var parts = [];
        var key;
        var i;
        if (value === null || kind === "undefined") { return "null"; }
        if (kind === "string") { return "\"" + escapeString(value) + "\""; }
        if (kind === "number") { return isFinite(value) ? String(value) : "null"; }
        if (kind === "boolean") { return value ? "true" : "false"; }
        if (value instanceof Array) {
            for (i = 0; i < value.length; i += 1) { parts.push(serialize(value[i])); }
            return "[" + parts.join(",") + "]";
        }
        for (key in value) {
            if (Object.prototype.hasOwnProperty.call(value, key) && typeof value[key] !== "function") {
                parts.push("\"" + escapeString(key) + "\":" + serialize(value[key]));
            }
        }
        return "{" + parts.join(",") + "}";
    }

    function writeReceipt(value) {
        if (receiptFile === null) { return; }
        requireCondition(!receiptFile.exists, "RECEIPT_ALREADY_EXISTS");
        receiptFile.encoding = "UTF-8";
        requireCondition(receiptFile.open("w"), "RECEIPT_OPEN_FAILED");
        try {
            requireCondition(receiptFile.write(serialize(value) + "\n"), "RECEIPT_WRITE_FAILED");
        } finally {
            receiptFile.close();
        }
    }

    function absoluteFile(path, extension, mustExist, field) {
        requireCondition(typeof path === "string" && path.length > 0, "INVALID_PATH_" + field);
        requireCondition(/^(?:[A-Za-z]:[\\\/]|\\\\|\/)/.test(path), "ABSOLUTE_PATH_REQUIRED_" + field);
        requireCondition(path.toLowerCase().slice(-extension.length) === extension, "WRONG_EXTENSION_" + field);
        var file = new File(path);
        requireCondition(file.parent.exists, "PARENT_FOLDER_MISSING_" + field);
        requireCondition(mustExist ? file.exists : !file.exists,
                         mustExist ? "INPUT_MISSING_" + field : "OUTPUT_ALREADY_EXISTS_" + field);
        return file;
    }

    function exactInteger(value, minimum, maximum, field) {
        requireCondition(typeof value === "number" && isFinite(value) && value === Math.floor(value)
                         && value >= minimum && value <= maximum, "INVALID_" + field);
    }

    function sequenceEvidence(firstFile, frameCount, field) {
        var first = /^(.*?)(\d+)(\.(?:png|exr))$/i.exec(firstFile.name);
        requireCondition(first !== null, "NUMBERED_PNG_OR_EXR_SEQUENCE_REQUIRED_" + field);
        var firstNumber = parseInt(first[2], 10);
        var files = firstFile.parent.getFiles();
        var found = {};
        var count = 0;
        var match;
        var number;
        var i;
        for (i = 0; i < files.length; i += 1) {
            if (!(files[i] instanceof File)) { continue; }
            match = /^(.*?)(\d+)(\.(?:png|exr))$/i.exec(files[i].name);
            if (match !== null && match[1] === first[1] && match[2].length === first[2].length
                    && match[3].toLowerCase() === first[3].toLowerCase()) {
                number = parseInt(match[2], 10);
                found[String(number)] = true;
                count += 1;
            }
        }
        requireCondition(count === frameCount, "SEQUENCE_FILE_COUNT_MISMATCH_" + field);
        for (i = 0; i < frameCount; i += 1) {
            requireCondition(found[String(firstNumber + i)] === true, "SEQUENCE_GAP_OR_WRONG_FIRST_FRAME_" + field);
        }
        return {first_path: firstFile.fsName, first_number: firstNumber,
                last_number: firstNumber + frameCount - 1, file_count: count};
    }

    function copyVector(value) {
        var result = [];
        var i;
        for (i = 0; i < value.length; i += 1) { result.push(Number(value[i])); }
        return result;
    }

    function transformProperty(layer, name) {
        return layer.property("ADBE Transform Group").property(name);
    }

    function closeNumber(actual, expected, epsilon, field) {
        requireCondition(typeof actual === "number" && isFinite(actual)
                         && Math.abs(actual - expected) <= epsilon, "READBACK_MISMATCH_" + field);
    }

    function alignedVector(actual, expectedX, expectedY, neutralZ, field) {
        requireCondition(actual.length === 2 || actual.length === 3, "INVALID_VECTOR_" + field);
        closeNumber(actual[0], expectedX, GEOMETRY_EPSILON, field + "_X");
        closeNumber(actual[1], expectedY, GEOMETRY_EPSILON, field + "_Y");
        if (actual.length === 3) { closeNumber(actual[2], neutralZ, GEOMETRY_EPSILON, field + "_Z"); }
    }

    function layerEvidence(layer, width, height, duration, fps) {
        var data = {
            id: layer.id,
            index: layer.index,
            name: layer.name,
            source_id: layer.source.id,
            threeDLayer: layer.threeDLayer,
            position: copyVector(transformProperty(layer, "ADBE Position").value),
            anchor: copyVector(transformProperty(layer, "ADBE Anchor Point").value),
            scale: copyVector(transformProperty(layer, "ADBE Scale").value),
            rotation: Number(transformProperty(layer, "ADBE Rotate Z").value),
            opacity: Number(transformProperty(layer, "ADBE Opacity").value),
            start_time: layer.startTime,
            in_point: layer.inPoint,
            out_point: layer.outPoint,
            stretch: layer.stretch,
            time_remap_enabled: layer.timeRemapEnabled,
            enabled: layer.enabled,
            parent_is_null: layer.parent === null
        };
        requireCondition(data.threeDLayer === false && data.parent_is_null, "LAYER_NOT_ALIGNED_2D_" + layer.name);
        alignedVector(data.position, width / 2, height / 2, 0, layer.name + "_POSITION");
        alignedVector(data.anchor, width / 2, height / 2, 0, layer.name + "_ANCHOR");
        alignedVector(data.scale, 100, 100, 100, layer.name + "_SCALE");
        closeNumber(data.rotation, 0, GEOMETRY_EPSILON, layer.name + "_ROTATION");
        closeNumber(data.opacity, 100, GEOMETRY_EPSILON, layer.name + "_OPACITY");
        closeNumber(data.stretch, 100, GEOMETRY_EPSILON, layer.name + "_STRETCH");
        closeNumber(data.start_time, 0, MAX_FRAME_RESIDUAL / fps, layer.name + "_START");
        closeNumber(data.in_point, 0, MAX_FRAME_RESIDUAL / fps, layer.name + "_IN");
        closeNumber(data.out_point, duration, MAX_FRAME_RESIDUAL / fps, layer.name + "_OUT");
        requireCondition(data.time_remap_enabled === false, "TIME_REMAP_UNEXPECTED_" + layer.name);
        return data;
    }

    function clockEvidence(item, count, fps, field) {
        var actualFps = item.frameRate;
        var actualDuration = item.duration;
        var maximumResidual = 0;
        var residual;
        var nativeIndex;
        var i;
        requireCondition(actualFps > 0 && isFinite(actualFps), "INVALID_NATIVE_FPS_" + field);
        closeNumber(actualFps, fps, Math.max(0.00001, fps * 0.000001), field + "_FPS");
        requireCondition(Math.round(actualDuration * actualFps) === count, "NATIVE_FRAME_COUNT_MISMATCH_" + field);
        closeNumber(actualDuration * fps, count, MAX_FRAME_RESIDUAL, field + "_DURATION_FRAMES");
        for (i = 0; i < count; i += 1) {
            nativeIndex = (i / fps) * actualFps;
            requireCondition(Math.round(nativeIndex) === i, "NATIVE_FRAME_MAPPING_MISMATCH_" + field);
            residual = Math.abs(nativeIndex - i);
            if (residual > maximumResidual) { maximumResidual = residual; }
        }
        requireCondition(maximumResidual <= MAX_FRAME_RESIDUAL, "NATIVE_FRAME_CLOCK_RESIDUAL_" + field);
        return {requested_fps: fps, native_fps: actualFps, native_frame_duration: item.frameDuration,
                duration: actualDuration, frame_count: count,
                duration_residual_frames: Math.abs(actualDuration * fps - count),
                maximum_frame_index_residual: maximumResidual,
                checked_integer_frame_indices: count};
    }

    function importSequence(file, name, folder, requireAlpha) {
        var options = new ImportOptions(file);
        requireCondition(options.canImportAs(ImportAsType.FOOTAGE), "CANNOT_IMPORT_FOOTAGE_" + name);
        options.importAs = ImportAsType.FOOTAGE;
        options.sequence = true;
        options.forceAlphabetical = false;
        var item = project.importFile(options);
        item.name = name;
        item.parentFolder = folder;
        requireCondition(item.width === CONFIG.width && item.height === CONFIG.height, "FOOTAGE_DIMENSIONS_MISMATCH_" + name);
        closeNumber(item.pixelAspect, 1, 0.000001, name + "_PIXEL_ASPECT");
        requireCondition(item.mainSource.isStill === false, "FOOTAGE_IMPORTED_AS_STILL_" + name);
        item.mainSource.conformFrameRate = CONFIG.fps;
        if (requireAlpha) { requireCondition(item.mainSource.hasAlpha === true, "ALPHA_CHANNEL_MISSING"); }
        if (item.mainSource.hasAlpha) {
            item.mainSource.alphaMode = AlphaMode.STRAIGHT;
            item.mainSource.invertAlpha = false;
            requireCondition(item.mainSource.alphaMode === AlphaMode.STRAIGHT && item.mainSource.invertAlpha === false,
                             "STRAIGHT_ALPHA_READBACK_FAILED_" + name);
        }
        return item;
    }

    function makeComp(name, width, height, folder, duration) {
        var comp = project.items.addComp(name, width, height, 1, duration, CONFIG.fps);
        comp.parentFolder = folder;
        comp.displayStartTime = 0;
        comp.workAreaStart = 0;
        comp.workAreaDuration = duration;
        comp.resolutionFactor = [1, 1];
        return comp;
    }

    function addLayer(comp, item, name, duration) {
        var layer = comp.layers.add(item);
        layer.name = name;
        layer.startTime = 0;
        layer.inPoint = 0;
        layer.outPoint = duration;
        layer.stretch = 100;
        layer.motionBlur = false;
        layer.frameBlendingType = FrameBlendingType.NO_FRAME_BLEND;
        requireCondition(layer.frameBlending === false, "FRAME_BLENDING_DISABLED_READBACK_FAILED");
        layer.quality = LayerQuality.BEST;
        return layer;
    }

    function addReviewLabel(comp, text, x, y, size, duration) {
        var layer = comp.layers.addText(text);
        var property = layer.property("ADBE Text Properties").property("ADBE Text Document");
        var document = property.value;
        document.fontSize = size;
        document.applyFill = true;
        document.fillColor = [1, 1, 1];
        document.applyStroke = false;
        property.setValue(document);
        transformProperty(layer, "ADBE Position").setValue([x, y]);
        layer.outPoint = duration;
    }

    function compEvidence(comp, count, fps) {
        return {id: comp.id, name: comp.name, width: comp.width, height: comp.height,
                pixel_aspect: comp.pixelAspect, layer_count: comp.numLayers,
                clock: clockEvidence(comp, count, fps, comp.name)};
    }

    function alphaComp(name, footage, folder, duration) {
        var comp = makeComp(name, CONFIG.width, CONFIG.height, folder, duration);
        addLayer(comp, footage, "WHITE RGB + REAL COVERAGE A (straight)", duration);
        return comp;
    }
    function makeRoto(name, rgbFootage, matteFootage, folder, duration) {
        requireCondition(rgbFootage.mainSource.hasAlpha === false, "RGB_INPUT_MUST_NOT_HAVE_A_" + name);
        requireCondition(matteFootage.mainSource.hasAlpha === true, "MATTE_INPUT_REQUIRES_REAL_A_" + name);
        var comp = makeComp(name, CONFIG.width, CONFIG.height, folder, duration);
        var rgb = addLayer(comp, rgbFootage, "RGB - ONE ALPHA TRACK MATTE", duration);
        var matte = addLayer(comp, matteFootage, "COVERAGE A - NO RGB UNMIX", duration);
        matte.moveBefore(rgb);
        requireCondition(typeof rgb.setTrackMatte === "function", "NATIVE_TRACK_MATTE_API_REQUIRED");
        rgb.setTrackMatte(matte, TrackMatteType.ALPHA);
        matte.enabled = false;
        rgb.enabled = true;
        requireCondition(rgb.hasTrackMatte === true && rgb.trackMatteType === TrackMatteType.ALPHA
                         && rgb.trackMatteLayer !== null && rgb.trackMatteLayer.id === matte.id,
                         "REAL_ALPHA_TRACK_MATTE_READBACK_FAILED_" + name);
        requireCondition(rgb.property("ADBE Effect Parade").numProperties === 0
                         && matte.property("ADBE Effect Parade").numProperties === 0,
                         "UNEXPECTED_EFFECT_ALPHA_TRANSFORM_" + name);
        receipt.track_mattes.push({comp: name, alpha_applications_in_branch: 1,
            rgb_input_has_alpha: false, real_alpha_readback: true, native_type: String(rgb.trackMatteType),
            rgb: layerEvidence(rgb, CONFIG.width, CONFIG.height, duration, CONFIG.fps),
            matte: layerEvidence(matte, CONFIG.width, CONFIG.height, duration, CONFIG.fps)});
        return comp;
    }
    function composite(name, roto, color, folder, duration) {
        var comp = makeComp(name, CONFIG.width, CONFIG.height, folder, duration);
        var bg = comp.layers.addSolid(color, "EDITABLE BACKGROUND", CONFIG.width, CONFIG.height, 1, duration);
        bg.moveToEnd();
        var cutout = addLayer(comp, roto, "ASSOCIATED CUTOUT - NO SECOND MATTE", duration);
        requireCondition(cutout.hasTrackMatte === false, "SECOND_COMPOSITE_MATTE_FORBIDDEN");
        if (!receipt.composite_branches) { receipt.composite_branches = []; }
        receipt.composite_branches.push({comp: name, second_track_matte: false});
        return comp;
    }
    try {
        requireCondition(CONFIG !== null && typeof CONFIG === "object", "INVALID_CONFIG");
        requireCondition(CONFIG.host_build_authorized === true, "PREPARATION_ONLY_HOST_BUILD_NOT_AUTHORIZED");
        requireCondition(CONFIG.full_input_validation_passed === true
                         && typeof CONFIG.input_validation_sha256 === "string"
                         && CONFIG.input_validation_sha256.length === 64, "FULL65_INPUT_VALIDATION_REQUIRED");
        exactInteger(CONFIG.width, 4096, 4096, "WIDTH");
        exactInteger(CONFIG.height, 2160, 2160, "HEIGHT");
        exactInteger(CONFIG.frame_count, 65, 65, "FRAME_COUNT");
        exactInteger(CONFIG.fps, 25, 25, "FPS");
        var duration = CONFIG.frame_count / CONFIG.fps;
        var sourceFile = absoluteFile(CONFIG.source_sequence_first, ".png", true, "SOURCE");
        var foregroundFile = absoluteFile(CONFIG.foreground_sequence_first, ".exr", true, "FOREGROUND");
        var alphaFile = absoluteFile(CONFIG.alpha_sequence_first, ".exr", true, "DERIVED_REFINED_A");
        var priorFile = absoluteFile(CONFIG.raw_prior_sequence_first, ".exr", true, "RAW_LTX_PRIOR_A");
        var projectFile = absoluteFile(CONFIG.project_path, ".aep", false, "NEW_PROJECT");
        receiptFile = absoluteFile(CONFIG.receipt_path, ".json", false, "NEW_RECEIPT");
        receipt.input_validation_path = CONFIG.input_validation_path;
        receipt.input_validation_sha256 = CONFIG.input_validation_sha256;
        receipt.refinement_method = CONFIG.refinement_method;
        receipt.app_version = String(app.version);
        requireCondition(parseInt(app.version, 10) === 26, "AE2026_VERSION_26_REQUIRED");
        project = app.project;
        receipt.initial_project = {present: project !== null, item_count: project ? project.numItems : 0,
            dirty: project ? project.dirty : false, has_saved_file: project ? project.file !== null : false};
        requireCondition(project === null || (project.numItems === 0 && project.dirty === false && project.file === null),
                         "EXISTING_USER_PROJECT_PRESERVED_EMPTY_CLEAN_PROJECT_REQUIRED");
        receipt.inputs = {source: sequenceEvidence(sourceFile, 65, "SOURCE"),
            foreground: sequenceEvidence(foregroundFile, 65, "FOREGROUND_RGB_NO_ALPHA"),
            derived_refined_A: sequenceEvidence(alphaFile, 65, "DERIVED_REFINED_A"),
            mapped_raw_LTX_prior_A: sequenceEvidence(priorFile, 65, "MAPPED_RAW_LTX_PRIOR_A")};
        if (project === null) { project = app.newProject(); }
        requireCondition(project !== null && project.numItems === 0 && project.dirty === false && project.file === null,
                         "EMPTY_CLEAN_PROJECT_REQUIRED");
        receipt.stage = "import_and_build";
        app.beginUndoGroup("4K refined alpha + restored RGB - editable native alpha mattes");
        undoOpen = true; receipt.project_changed = true;
        project.bitsPerChannel = 32;
        project.workingSpace = "";
        project.linearBlending = false;
        project.linearizeWorkingSpace = false;
        requireCondition(project.bitsPerChannel === 32 && (project.workingSpace === "" || project.workingSpace === "None")
            && project.linearBlending === false && project.linearizeWorkingSpace === false,
            "NATIVE_32BPC_UNMANAGED_CODEVALUE_READBACK_REQUIRED");
        var footageFolder = project.items.addFolder("01_FOOTAGE - RGB AND ALPHA SEPARATE");
        var alphaFolder = project.items.addFolder("02_ALPHA - RAW PRIOR AND DERIVED REFINED");
        var rotoFolder = project.items.addFolder("03_EDITABLE_ALPHA_TRACK_MATTES");
        var compositeFolder = project.items.addFolder("04_COMPOSITES");
        var reviewFolder = project.items.addFolder("05_FOUR_BACKGROUND_REVIEW");
        var source = importSequence(sourceFile, "SOURCE_4K_RGB8_CODED_BT709", footageFolder, false);
        var foreground = importSequence(foregroundFile, "RESTORED_FOREGROUND_F32_CODED_RGB_NO_A", footageFolder, false);
        var alpha = importSequence(alphaFile, "DERIVED_REFINED_F32_COVERAGE_A_WHITE_RGB_STRAIGHT", footageFolder, true);
        var prior = importSequence(priorFile, "MAPPED_RAW_LTX_F32_PRIOR_A_WHITE_RGB_STRAIGHT", footageFolder, true);
        requireCondition(source.mainSource.hasAlpha === false && foreground.mainSource.hasAlpha === false,
                         "RGB_INPUT_A_MUST_BE_ABSENT_TO_PREVENT_A_SQUARED");
        var imported = [source, foreground, alpha, prior];
        receipt.clocks = [];
        for (var i = 0; i < imported.length; i += 1) {
            receipt.clocks.push({name: imported[i].name, clock: clockEvidence(imported[i], 65, 25, imported[i].name),
                has_alpha: imported[i].mainSource.hasAlpha,
                alpha_mode_raw: imported[i].mainSource.hasAlpha ? String(imported[i].mainSource.alphaMode) : null});
        }
        var sourceComp = makeComp("SOURCE_4K", 4096, 2160, compositeFolder, duration);
        addLayer(sourceComp, source, "ORIGINAL RGB8 CODE VALUES", duration);
        var fComp = makeComp("RESTORED_FOREGROUND_RGB_4K", 4096, 2160, compositeFolder, duration);
        addLayer(fComp, foreground, "UNASSOCIATED F32 CODED RGB - NO A", duration);
        alphaComp("RAW_LTX_PRIOR_ALPHA_4K", prior, alphaFolder, duration);
        alphaComp("DERIVED_REFINED_ALPHA_4K", alpha, alphaFolder, duration);
        receipt.track_mattes = [];
        var rawRoto = makeRoto("ROTO_RAW_PRIOR_4K", source, prior, rotoFolder, duration);
        var derivedRoto = makeRoto("ROTO_SOURCE_REFINED_4K", source, alpha, rotoFolder, duration);
        var cleanRoto = makeRoto("ROTO_CLEAN_REFINED_4K", foreground, alpha, rotoFolder, duration);
        var neutral = [0.32, 0.32, 0.32];
        composite("COMPOSITE_RAW_PRIOR_4K", rawRoto, neutral, compositeFolder, duration);
        composite("COMPOSITE_SOURCE_REFINED_4K", derivedRoto, neutral, compositeFolder, duration);
        composite("COMPOSITE_CLEAN_REFINED_4K", cleanRoto, neutral, compositeFolder, duration);
        var backgrounds = [{name:"BLACK",rgb:[0,0,0]}, {name:"GRAY",rgb:neutral},
            {name:"WHITE",rgb:[1,1,1]}, {name:"CONTRAST_RED",rgb:[0.65,0.055,0.075]}];
        var variants = [{name:"RAW_PRIOR",comp:rawRoto}, {name:"SOURCE_REFINED",comp:derivedRoto}, {name:"CLEAN_REFINED",comp:cleanRoto}];
        receipt.review_comps = [];
        for (var b = 0; b < backgrounds.length; b += 1) {
            for (var v = 0; v < variants.length; v += 1) {
                var review = composite("REVIEW_" + variants[v].name + "_" + backgrounds[b].name,
                    variants[v].comp, backgrounds[b].rgb, reviewFolder, duration);
                receipt.review_comps.push(compEvidence(review, 65, 25));
            }
        }
        receipt.source_clock = clockEvidence(sourceComp, 65, 25, "SOURCE_4K");
        receipt.precision_contract = "F32 raw mapped LTX prior is preserved separately; derived/refined A is a distinct matte. Restored foreground EXR has RGB only. Every roto branch applies its real ALPHA track matte exactly once. No cleanup effect, levels, blur, feather, erosion, resize or color conversion is added in AE.";
        receipt.code_value_color_contract = "Unmanaged32bpc; source BT709 code values and restored coded RGB; A linear coverage. Native export must use Preserve RGB, Convert To Linear Light Off, ZIP/FLOAT32. Template and actual output headers/pixels require separate host validation.";
        requireCondition(project.renderQueue.numItems === 0, "BUILDER_MUST_NOT_ADD_RENDER_ITEMS");
        app.endUndoGroup(); undoOpen = false;
        receipt.stage = "save_new_owned_project";
        requireCondition(!projectFile.exists, "EXISTING_PROJECT_OUTPUT_PRESERVED");
        project.save(projectFile);
        requireCondition(projectFile.exists && project.file !== null && project.file.fsName === projectFile.fsName
                         && project.dirty === false, "PROJECT_SAVE_READBACK_FAILED");
        receipt.project_saved = true;
        receipt.project_path = project.file.fsName;
        receipt.project = {bits_per_channel:project.bitsPerChannel, working_space_raw:project.workingSpace,
            working_gamma_raw:project.workingGamma, color_management_system_raw:project.colorManagementSystem,
            linear_blending:project.linearBlending, linearize_working_space:project.linearizeWorkingSpace,
            item_count:project.numItems, render_queue_count:project.renderQueue.numItems};
        receipt.foreground_input_interpretation = {verified:false, required:"Interpret Footage > Color > Preserve RGB on coded foreground EXR; native original F*A probe must pass before full queue"};
        receipt.native_output_validation = "PENDING_INPUT_INTERPRETATION_AND_NATIVE_PROBE";
        receipt.visual_approval = false;
        receipt.ok = true;
        receipt.stage = "BUILT_SAVED_NATIVE_4K_VALIDATION_REQUIRED";
        receipt.completed_at_utc = new Date().toUTCString();
        writeReceipt(receipt);
        return serialize(receipt);
    } catch (error) {
        receipt.ok = false; receipt.error = String(error); receipt.failed_at_utc = new Date().toUTCString();
        if (receiptFile !== null && !receiptFile.exists) { try { writeReceipt(receipt); } catch (ignored) {} }
        throw new Error("LTX_4K_REFINED_BUILD_FAILED [" + receipt.stage + "]: " + String(error));
    } finally { if (undoOpen) { app.endUndoGroup(); } }
}());
