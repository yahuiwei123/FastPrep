// Explicit DSL2 graph corresponding to conf/workflow.json.
include { PREFLIGHT; ASSEMBLE; COLLECT_QC } from './modules/stages'
include { RUN_NODE as ANAT_GET_T1W_FILE_IN_BIDS } from './modules/stages'
include { RUN_QC as ANAT_INPUT_QC } from './modules/stages'
include { RUN_NODE as ANAT_CREATE_SUBJECT_ORIG_DIR } from './modules/stages'
include { RUN_NODE as SESSION_ALIGN_COMPANIONS } from './modules/stages'
include { RUN_NODE as SESSION_NORMALIZE_MNI } from './modules/stages'
include { RUN_NODE as ANAT_MOTIONCOR } from './modules/stages'
include { RUN_QC as SESSION_ALIGNMENT_QC } from './modules/stages'
include { RUN_NODE as ANAT_ORIG_TO_NIFTI } from './modules/stages'
include { RUN_NODE as ANAT_SYNTHSEG } from './modules/stages'
include { RUN_NODE as ANAT_HDBET_MASK } from './modules/stages'
include { RUN_NODE as ANAT_MASK_TO_MGZ } from './modules/stages'
include { RUN_QC as ANAT_SEGMENTATION_QC } from './modules/stages'
include { RUN_NODE as ANAT_PREPARE_SYNTHSEG_ASEG } from './modules/stages'
include { RUN_NODE as ANAT_TISSUE_BIAS_CORRECT } from './modules/stages'
include { RUN_NODE as ANAT_GCA_REGISTER } from './modules/stages'
include { RUN_NODE as ANAT_TALAIRACH_AND_NU } from './modules/stages'
include { RUN_NODE as ANAT_BRAINMASK } from './modules/stages'
include { RUN_QC as ANAT_PREPROC_QC } from './modules/stages'
include { RUN_NODE as ANAT_PAINT_CC_TO_ASEG } from './modules/stages'
include { RUN_NODE as ANAT_SYNTHSEG_FILL } from './modules/stages'
include { RUN_QC as ANAT_VOLUME_SUPPORT_QC } from './modules/stages'
include { RUN_NODE as ANAT_TESSELLATE } from './modules/stages'
include { RUN_SURFACE as ANAT_FIX_TOPOLOGY } from './modules/stages'
include { RUN_NODE as ANAT_REMOVE_INTERSECTION } from './modules/stages'
include { RUN_NODE as ANAT_AUTODET_GWSTATS } from './modules/stages'
include { RUN_QC as ANAT_SURFACE_INIT_QC } from './modules/stages'
include { RUN_SURFACE as ANAT_WHITE_SURFACE } from './modules/stages'
include { RUN_NODE as ANAT_CORTEX_HIPAMYG_LABEL } from './modules/stages'
include { RUN_NODE as ANAT_CORTEX_LABEL } from './modules/stages'
include { RUN_NODE as ANAT_HYPORELABEL } from './modules/stages'
include { RUN_QC as ANAT_WHITE_CORTEX_QC } from './modules/stages'
include { RUN_NODE as ANAT_SMOOTH_INFLATED } from './modules/stages'
include { RUN_NODE as ANAT_CURV_STATS } from './modules/stages'
include { RUN_NODE as ANAT_SPHERE } from './modules/stages'
include { RUN_NODE as ANAT_SPHERE_REGISTER } from './modules/stages'
include { RUN_NODE as ANAT_CORTPARC_APARC } from './modules/stages'
include { RUN_NODE as ANAT_JACOBIAN } from './modules/stages'
include { RUN_QC as ANAT_SPHERICAL_REGISTRATION_QC } from './modules/stages'
include { RUN_SURFACE as ANAT_PIAL_SURFACE } from './modules/stages'
include { RUN_NODE as ANAT_RIBBON } from './modules/stages'
include { RUN_NODE as ANAT_CORTICAL_PAIR } from './modules/stages'
include { RUN_QC as ANAT_PIAL_RIBBON_QC } from './modules/stages'
include { RUN_NODE as ANAT_APAS2ASEG } from './modules/stages'
include { RUN_NODE as ANAT_PARCSTATS } from './modules/stages'
include { RUN_NODE as ANAT_PCTSURFCON } from './modules/stages'
include { RUN_NODE as VALIDATE_FSAVERAGE6_SPHERE_CORRESPONDENCE } from './modules/stages'
include { RUN_NODE as ANAT_APARC2ASEG } from './modules/stages'
include { RUN_NODE as RESAMPLE_WHITE_SURFACE_FSAVERAGE6 } from './modules/stages'
include { RUN_QC as ANAT_NATIVE_OUTPUTS_QC } from './modules/stages'
include { RUN_NODE as RESAMPLE_PIAL_SURFACE_FSAVERAGE6 } from './modules/stages'
include { RUN_QC as ANAT_FSAVERAGE6_QC } from './modules/stages'

def mergeLineage(left, right) {
    left.each { identity, directory ->
        if (right.containsKey(identity) && directory.toString() != right[identity].toString()) {
            error("Conflicting ancestor result for ${identity}")
        }
    }
    return left + right
}

def joinLineages(left, right) {
    return left.join(right, by: 0, failOnDuplicate: true, failOnMismatch: !params.isolate_session_errors)
        .map { subject, first, second -> tuple(subject, mergeLineage(first, second)) }
}

def stageInput(source) {
    return source.map { subject, lineage -> tuple(subject, lineage, lineage.values().toList()) }
}

workflow {
    main:
    if (params.help) {
        log.info('Use: nextflow run main.nf --runtime control/runtime.json --blueprint control/workflow.json --helper_python /path/to/helper/python --outdir results')
        log.info('T1 white/pial placement: --t1_placement_backend cpu|checked|gpu (default: gpu)')
        log.info('T2 pial refinement: --t2_pial_backend gpu|checked|cpu (default: gpu; no CPU replay)')
        log.info('Topology quick-sphere geometry: --topology_backend gpu|cpu (default: gpu; GA remains native)')
        log.info('QC mode: --qc_mode report_only|strict (default: report_only). Measured quality defects are reported and processing continues when scientific outputs are usable.')
        log.info('QC surface rendering: --qc_surface_engine reuse|reference (default: reuse)')
        log.info('Topology test: nextflow run main.nf -profile topology -stub-run --outdir topology-results')
    } else {
        if (!params.runtime) { error('Required: --runtime from bin/prepare.py; see README.md') }
        def destination = file(params.outdir).toFile()
        if (destination.exists() && !workflow.resume && destination.listFiles()?.size()) {
            error('Use a new empty --outdir, or -resume for the same inputs and configuration')
        }
        if (!(params.threads.toString() ==~ /[1-9][0-9]*/)) { error('--threads must be a positive integer') }
        if (!(params.qc_mode in ['strict', 'report_only'])) { error('--qc_mode must be strict or report_only') }
        if (!(params.qc_surface_engine in ['reuse', 'reference'])) { error('--qc_surface_engine must be reuse or reference') }
        if (!(params.t1_placement_backend in ['cpu', 'checked', 'gpu'])) { error('--t1_placement_backend must be cpu, checked or gpu') }
        if (!(params.t2_pial_backend in ['cpu', 'checked', 'gpu'])) { error('--t2_pial_backend must be gpu, checked or cpu') }
        if (!(params.topology_backend in ['gpu', 'cpu'])) { error('--topology_backend must be gpu or cpu') }
        if (params.medial_wall_strategy != null && !(params.medial_wall_strategy in ['legacy-final-pin', 'fixed-during-placement', 'fixed-on-intersection'])) {
            error('--medial_wall_strategy must be legacy-final-pin, fixed-during-placement or fixed-on-intersection')
        }
        if (params.qc_mode == 'report_only') { log.info('QC report_only: quality findings are recorded with usable outputs. Visual review is pending.') }
        def helpers = channel.value(file("${projectDir}/bin", checkIfExists: true))
        // Retain the original helpers input for unaffected science tasks and strict runs.
        def reportHelpers = params.qc_mode == 'report_only' ? channel.value(file("${projectDir}/report_only/bin", checkIfExists: true)) : helpers
        def motioncorHelpers = params.motioncor_helpers ? channel.value(file(params.motioncor_helpers, checkIfExists: true)) : helpers
        def mniHelpers = params.mni_helpers ? channel.value(file(params.mni_helpers, checkIfExists: true)) : reportHelpers
        def sphereHelpers = params.sphere_helpers ? channel.value(file(params.sphere_helpers, checkIfExists: true)) : reportHelpers
        def pialHelpers = params.pial_helpers ? channel.value(file(params.pial_helpers, checkIfExists: true)) : reportHelpers
        def whiteHelpers = params.white_helpers ? channel.value(file(params.white_helpers, checkIfExists: true)) : reportHelpers
        def whiteRuntime = channel.value(file(params.white_runtime ?: "${projectDir}/vendor/t2_pial", checkIfExists: true))
        def initializationHelpers = params.initialization_helpers ? channel.value(file(params.initialization_helpers, checkIfExists: true)) : reportHelpers
        def intersectionHelpers = params.intersection_helpers ? channel.value(file(params.intersection_helpers, checkIfExists: true)) : initializationHelpers
        def pialRuntime = channel.value(file(params.pial_runtime ?: "${projectDir}/vendor/t2_pial", checkIfExists: true))
        def pialBackends = [t1_placement_backend: params.t1_placement_backend, t2_pial_backend: params.t2_pial_backend]
        if (params.medial_wall_strategy != null) { pialBackends.medial_wall_strategy = params.medial_wall_strategy }
        PREFLIGHT(channel.value(file(params.runtime, checkIfExists: true)),
                  channel.value(file(params.blueprint, checkIfExists: true)), helpers, params.helper_python)
        def runtime = PREFLIGHT.out.runtime
        def blueprint = PREFLIGHT.out.blueprint
        def initializationBlueprint = params.initialization_blueprint ? channel.value(file(params.initialization_blueprint, checkIfExists: true)) : blueprint
        def subjects = PREFLIGHT.out.subjects.splitText().map { subject -> tuple(subject.trim(), [:]) }
        ANAT_GET_T1W_FILE_IN_BIDS(stageInput(subjects), runtime, blueprint, helpers,
            params.helper_python, 'anat_get_t1w_file_in_bids', params.threads.toString().toInteger(), params.device)
        def ch_anat_get_t1w_file_in_bids = ANAT_GET_T1W_FILE_IN_BIDS.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_get_t1w_file_in_bids'): result])
        }

        def input_anat_input_qc = ch_anat_get_t1w_file_in_bids
        ANAT_INPUT_QC(stageInput(input_anat_input_qc), runtime, blueprint, reportHelpers,
            params.helper_python, 'anat_input_qc', params.qc_mode, params.qc_surface_engine)
        def completed_anat_input_qc = ANAT_INPUT_QC.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_input_qc'): result])
        }
        def ch_anat_input_qc = params.qc_mode == 'strict' ? completed_anat_input_qc : input_anat_input_qc

        ANAT_CREATE_SUBJECT_ORIG_DIR(stageInput(ch_anat_input_qc), runtime, blueprint, helpers,
            params.helper_python, 'anat_create_subject_orig_dir', params.threads.toString().toInteger(), params.device)
        def ch_anat_create_subject_orig_dir = ANAT_CREATE_SUBJECT_ORIG_DIR.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_create_subject_orig_dir'): result])
        }

        SESSION_ALIGN_COMPANIONS(stageInput(ch_anat_input_qc), runtime, blueprint, helpers,
            params.helper_python, 'session_align_companions', params.threads.toString().toInteger(), params.device)
        def ch_session_align_companions = SESSION_ALIGN_COMPANIONS.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('session_align_companions'): result])
        }
        SESSION_NORMALIZE_MNI(stageInput(ch_session_align_companions), runtime, blueprint, mniHelpers,
            params.helper_python, 'session_normalize_mni', params.threads.toString().toInteger(), params.device)
        def ch_session_normalize_mni = SESSION_NORMALIZE_MNI.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('session_normalize_mni'): result])
        }

        ANAT_MOTIONCOR(stageInput(ch_anat_create_subject_orig_dir), runtime, blueprint, motioncorHelpers,
            params.helper_python, 'anat_motioncor', params.threads.toString().toInteger(), params.device)
        def ch_anat_motioncor = ANAT_MOTIONCOR.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_motioncor'): result])
        }

        def input_session_alignment_qc = ch_session_align_companions
        SESSION_ALIGNMENT_QC(stageInput(input_session_alignment_qc), runtime, blueprint, reportHelpers,
            params.helper_python, 'session_alignment_qc', params.qc_mode, params.qc_surface_engine)
        def completed_session_alignment_qc = SESSION_ALIGNMENT_QC.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('session_alignment_qc'): result])
        }
        def ch_session_alignment_qc = params.qc_mode == 'strict' ? completed_session_alignment_qc : input_session_alignment_qc

        ANAT_ORIG_TO_NIFTI(stageInput(ch_anat_motioncor), runtime, blueprint, helpers,
            params.helper_python, 'anat_orig_to_nifti', params.threads.toString().toInteger(), params.device)
        def ch_anat_orig_to_nifti = ANAT_ORIG_TO_NIFTI.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_orig_to_nifti'): result])
        }

        ANAT_SYNTHSEG(stageInput(ch_anat_motioncor), runtime, blueprint, helpers,
            params.helper_python, 'anat_synthseg', params.threads.toString().toInteger(), params.device)
        def ch_anat_synthseg = ANAT_SYNTHSEG.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_synthseg'): result])
        }

        ANAT_HDBET_MASK(stageInput(ch_anat_orig_to_nifti), runtime, blueprint, helpers,
            params.helper_python, 'anat_hdbet_mask', params.threads.toString().toInteger(), params.device)
        def ch_anat_hdbet_mask = ANAT_HDBET_MASK.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_hdbet_mask'): result])
        }

        ANAT_MASK_TO_MGZ(stageInput(joinLineages(ch_anat_hdbet_mask, ch_anat_motioncor)), runtime, blueprint, helpers,
            params.helper_python, 'anat_mask_to_mgz', params.threads.toString().toInteger(), params.device)
        def ch_anat_mask_to_mgz = ANAT_MASK_TO_MGZ.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_mask_to_mgz'): result])
        }

        def input_anat_segmentation_qc = joinLineages(joinLineages(joinLineages(joinLineages(ch_anat_hdbet_mask, ch_anat_mask_to_mgz), ch_anat_motioncor), ch_anat_orig_to_nifti), ch_anat_synthseg)
        ANAT_SEGMENTATION_QC(stageInput(input_anat_segmentation_qc), runtime, blueprint, reportHelpers,
            params.helper_python, 'anat_segmentation_qc', params.qc_mode, params.qc_surface_engine)
        def completed_anat_segmentation_qc = ANAT_SEGMENTATION_QC.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_segmentation_qc'): result])
        }
        def ch_anat_segmentation_qc = params.qc_mode == 'strict' ? completed_anat_segmentation_qc : input_anat_segmentation_qc

        ANAT_PREPARE_SYNTHSEG_ASEG(stageInput(ch_anat_segmentation_qc), runtime, blueprint, helpers,
            params.helper_python, 'anat_prepare_synthseg_aseg', params.threads.toString().toInteger(), params.device)
        def ch_anat_prepare_synthseg_aseg = ANAT_PREPARE_SYNTHSEG_ASEG.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_prepare_synthseg_aseg'): result])
        }

        ANAT_TISSUE_BIAS_CORRECT(stageInput(joinLineages(joinLineages(ch_anat_motioncor, ch_anat_prepare_synthseg_aseg), ch_anat_segmentation_qc)), runtime, blueprint, helpers,
            params.helper_python, 'anat_tissue_bias_correct', params.threads.toString().toInteger(), params.device)
        def ch_anat_tissue_bias_correct = ANAT_TISSUE_BIAS_CORRECT.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_tissue_bias_correct'): result])
        }

        ANAT_GCA_REGISTER(stageInput(joinLineages(ch_anat_segmentation_qc, ch_anat_tissue_bias_correct)), runtime, blueprint, helpers,
            params.helper_python, 'anat_gca_register', params.threads.toString().toInteger(), params.device)
        def ch_anat_gca_register = ANAT_GCA_REGISTER.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_gca_register'): result])
        }

        ANAT_TALAIRACH_AND_NU(stageInput(joinLineages(joinLineages(ch_anat_gca_register, ch_anat_motioncor), ch_anat_tissue_bias_correct)), runtime, blueprint, helpers,
            params.helper_python, 'anat_talairach_and_nu', params.threads.toString().toInteger(), params.device)
        def ch_anat_talairach_and_nu = ANAT_TALAIRACH_AND_NU.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_talairach_and_nu'): result])
        }

        ANAT_BRAINMASK(stageInput(joinLineages(ch_anat_segmentation_qc, ch_anat_talairach_and_nu)), runtime, blueprint, helpers,
            params.helper_python, 'anat_brainmask', params.threads.toString().toInteger(), params.device)
        def ch_anat_brainmask = ANAT_BRAINMASK.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_brainmask'): result])
        }

        def input_anat_preproc_qc = joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(ch_anat_brainmask, ch_anat_gca_register), ch_anat_motioncor), ch_anat_prepare_synthseg_aseg), ch_anat_segmentation_qc), ch_anat_talairach_and_nu), ch_anat_tissue_bias_correct)
        ANAT_PREPROC_QC(stageInput(input_anat_preproc_qc), runtime, blueprint, reportHelpers,
            params.helper_python, 'anat_preproc_qc', params.qc_mode, params.qc_surface_engine)
        def completed_anat_preproc_qc = ANAT_PREPROC_QC.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_preproc_qc'): result])
        }
        def ch_anat_preproc_qc = params.qc_mode == 'strict' ? completed_anat_preproc_qc : input_anat_preproc_qc

        ANAT_PAINT_CC_TO_ASEG(stageInput(joinLineages(joinLineages(ch_anat_prepare_synthseg_aseg, ch_anat_preproc_qc), ch_anat_segmentation_qc)), runtime, blueprint, helpers,
            params.helper_python, 'anat_paint_cc_to_aseg', params.threads.toString().toInteger(), params.device)
        def ch_anat_paint_cc_to_aseg = ANAT_PAINT_CC_TO_ASEG.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_paint_cc_to_aseg'): result])
        }

        ANAT_SYNTHSEG_FILL(stageInput(joinLineages(joinLineages(ch_anat_paint_cc_to_aseg, ch_anat_preproc_qc), ch_anat_talairach_and_nu)), runtime, blueprint, reportHelpers,
            params.helper_python, 'anat_synthseg_fill', params.threads.toString().toInteger(), params.device)
        def ch_anat_synthseg_fill = ANAT_SYNTHSEG_FILL.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_synthseg_fill'): result])
        }

        def input_anat_volume_support_qc = joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(ch_anat_paint_cc_to_aseg, ch_anat_prepare_synthseg_aseg), ch_anat_preproc_qc), ch_anat_segmentation_qc), ch_anat_synthseg_fill), ch_anat_talairach_and_nu)
        ANAT_VOLUME_SUPPORT_QC(stageInput(input_anat_volume_support_qc), runtime, blueprint, reportHelpers,
            params.helper_python, 'anat_volume_support_qc', params.qc_mode, params.qc_surface_engine)
        def completed_anat_volume_support_qc = ANAT_VOLUME_SUPPORT_QC.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_volume_support_qc'): result])
        }
        def ch_anat_volume_support_qc = params.qc_mode == 'strict' ? completed_anat_volume_support_qc : input_anat_volume_support_qc

        ANAT_TESSELLATE(stageInput(joinLineages(ch_anat_motioncor, ch_anat_volume_support_qc)), runtime, initializationBlueprint, initializationHelpers,
            params.helper_python, 'anat_tessellate', params.threads.toString().toInteger(), params.device)
        def ch_anat_tessellate = ANAT_TESSELLATE.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_tessellate'): result])
        }

        ANAT_FIX_TOPOLOGY(stageInput(joinLineages(joinLineages(ch_anat_motioncor, ch_anat_tessellate), ch_anat_volume_support_qc)), runtime, initializationBlueprint, initializationHelpers,
            params.helper_python, 'anat_fix_topology', params.threads.toString().toInteger(), params.device, [topology_backend: params.topology_backend],
            channel.value(file("${projectDir}/vendor/topology", checkIfExists: true)))
        def ch_anat_fix_topology = ANAT_FIX_TOPOLOGY.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_fix_topology'): result])
        }

        ANAT_REMOVE_INTERSECTION(stageInput(joinLineages(ch_anat_fix_topology, ch_anat_tessellate)), runtime, initializationBlueprint, intersectionHelpers,
            params.helper_python, 'anat_remove_intersection', params.threads.toString().toInteger(), params.device)
        def ch_anat_remove_intersection = ANAT_REMOVE_INTERSECTION.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_remove_intersection'): result])
        }

        ANAT_AUTODET_GWSTATS(stageInput(joinLineages(ch_anat_remove_intersection, ch_anat_volume_support_qc)), runtime, blueprint, helpers,
            params.helper_python, 'anat_autodet_gwstats', params.threads.toString().toInteger(), params.device)
        def ch_anat_autodet_gwstats = ANAT_AUTODET_GWSTATS.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_autodet_gwstats'): result])
        }

        def input_anat_surface_init_qc = joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(ch_anat_autodet_gwstats, ch_anat_fix_topology), ch_anat_motioncor), ch_anat_remove_intersection), ch_anat_tessellate), ch_anat_volume_support_qc)
        ANAT_SURFACE_INIT_QC(stageInput(input_anat_surface_init_qc), runtime, initializationBlueprint, initializationHelpers,
            params.helper_python, 'anat_surface_init_qc', params.qc_mode, params.qc_surface_engine)
        def completed_anat_surface_init_qc = ANAT_SURFACE_INIT_QC.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_surface_init_qc'): result])
        }
        def ch_anat_surface_init_qc = params.qc_mode == 'strict' ? completed_anat_surface_init_qc : input_anat_surface_init_qc

        ANAT_WHITE_SURFACE(stageInput(joinLineages(ch_anat_surface_init_qc, ch_anat_volume_support_qc)), runtime, blueprint, whiteHelpers,
            params.helper_python, 'anat_white_surface', params.threads.toString().toInteger(), params.device, [t1_placement_backend: params.t1_placement_backend],
            whiteRuntime)
        def ch_anat_white_surface = ANAT_WHITE_SURFACE.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_white_surface'): result])
        }

        ANAT_CORTEX_HIPAMYG_LABEL(stageInput(joinLineages(ch_anat_volume_support_qc, ch_anat_white_surface)), runtime, blueprint, helpers,
            params.helper_python, 'anat_cortex_hipamyg_label', params.threads.toString().toInteger(), params.device)
        def ch_anat_cortex_hipamyg_label = ANAT_CORTEX_HIPAMYG_LABEL.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_cortex_hipamyg_label'): result])
        }

        ANAT_CORTEX_LABEL(stageInput(joinLineages(ch_anat_volume_support_qc, ch_anat_white_surface)), runtime, blueprint, helpers,
            params.helper_python, 'anat_cortex_label', params.threads.toString().toInteger(), params.device)
        def ch_anat_cortex_label = ANAT_CORTEX_LABEL.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_cortex_label'): result])
        }

        ANAT_HYPORELABEL(stageInput(joinLineages(ch_anat_volume_support_qc, ch_anat_white_surface)), runtime, blueprint, helpers,
            params.helper_python, 'anat_hyporelabel', params.threads.toString().toInteger(), params.device)
        def ch_anat_hyporelabel = ANAT_HYPORELABEL.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_hyporelabel'): result])
        }

        def input_anat_white_cortex_qc = joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(ch_anat_cortex_hipamyg_label, ch_anat_cortex_label), ch_anat_hyporelabel), ch_anat_surface_init_qc), ch_anat_volume_support_qc), ch_anat_white_surface)
        ANAT_WHITE_CORTEX_QC(stageInput(input_anat_white_cortex_qc), runtime, blueprint, reportHelpers,
            params.helper_python, 'anat_white_cortex_qc', params.qc_mode, params.qc_surface_engine)
        def completed_anat_white_cortex_qc = ANAT_WHITE_CORTEX_QC.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_white_cortex_qc'): result])
        }
        def ch_anat_white_cortex_qc = params.qc_mode == 'strict' ? completed_anat_white_cortex_qc : input_anat_white_cortex_qc

        ANAT_SMOOTH_INFLATED(stageInput(ch_anat_white_cortex_qc), runtime, blueprint, helpers,
            params.helper_python, 'anat_smooth_inflated', params.threads.toString().toInteger(), params.device)
        def ch_anat_smooth_inflated = ANAT_SMOOTH_INFLATED.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_smooth_inflated'): result])
        }

        ANAT_CURV_STATS(stageInput(joinLineages(ch_anat_smooth_inflated, ch_anat_white_cortex_qc)), runtime, blueprint, helpers,
            params.helper_python, 'anat_curv_stats', params.threads.toString().toInteger(), params.device)
        def ch_anat_curv_stats = ANAT_CURV_STATS.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_curv_stats'): result])
        }

        ANAT_SPHERE(stageInput(ch_anat_smooth_inflated), runtime, blueprint, sphereHelpers,
            params.helper_python, 'anat_sphere', params.threads.toString().toInteger(), params.device)
        def ch_anat_sphere = ANAT_SPHERE.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_sphere'): result])
        }

        ANAT_SPHERE_REGISTER(stageInput(joinLineages(joinLineages(ch_anat_smooth_inflated, ch_anat_sphere), ch_anat_white_cortex_qc)), runtime, blueprint, sphereHelpers,
            params.helper_python, 'anat_sphere_register', params.threads.toString().toInteger(), params.device)
        def ch_anat_sphere_register = ANAT_SPHERE_REGISTER.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_sphere_register'): result])
        }

        ANAT_CORTPARC_APARC(stageInput(joinLineages(joinLineages(joinLineages(ch_anat_smooth_inflated, ch_anat_sphere_register), ch_anat_volume_support_qc), ch_anat_white_cortex_qc)), runtime, blueprint, helpers,
            params.helper_python, 'anat_cortparc_aparc', params.threads.toString().toInteger(), params.device)
        def ch_anat_cortparc_aparc = ANAT_CORTPARC_APARC.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_cortparc_aparc'): result])
        }

        ANAT_JACOBIAN(stageInput(joinLineages(ch_anat_sphere_register, ch_anat_white_cortex_qc)), runtime, blueprint, helpers,
            params.helper_python, 'anat_jacobian', params.threads.toString().toInteger(), params.device)
        def ch_anat_jacobian = ANAT_JACOBIAN.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_jacobian'): result])
        }

        def input_anat_spherical_registration_qc = joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(ch_anat_cortparc_aparc, ch_anat_curv_stats), ch_anat_jacobian), ch_anat_smooth_inflated), ch_anat_sphere), ch_anat_sphere_register), ch_anat_volume_support_qc), ch_anat_white_cortex_qc)
        ANAT_SPHERICAL_REGISTRATION_QC(stageInput(input_anat_spherical_registration_qc), runtime, blueprint, reportHelpers,
            params.helper_python, 'anat_spherical_registration_qc', params.qc_mode, params.qc_surface_engine)
        def completed_anat_spherical_registration_qc = ANAT_SPHERICAL_REGISTRATION_QC.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_spherical_registration_qc'): result])
        }
        def ch_anat_spherical_registration_qc = params.qc_mode == 'strict' ? completed_anat_spherical_registration_qc : input_anat_spherical_registration_qc

        ANAT_PIAL_SURFACE(stageInput(joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(ch_anat_segmentation_qc, ch_anat_spherical_registration_qc), ch_anat_surface_init_qc), ch_anat_volume_support_qc), ch_anat_white_cortex_qc), ch_session_alignment_qc)), runtime, blueprint, pialHelpers,
            params.helper_python, 'anat_pial_surface', params.threads.toString().toInteger(), params.device,
            pialBackends,
            pialRuntime)
        def ch_anat_pial_surface = ANAT_PIAL_SURFACE.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_pial_surface'): result])
        }

        ANAT_RIBBON(stageInput(joinLineages(joinLineages(ch_anat_pial_surface, ch_anat_volume_support_qc), ch_anat_white_cortex_qc)), runtime, blueprint, helpers,
            params.helper_python, 'anat_ribbon', params.threads.toString().toInteger(), params.device)
        def ch_anat_ribbon = ANAT_RIBBON.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_ribbon'): result])
        }

        ANAT_CORTICAL_PAIR(stageInput(ch_anat_ribbon), runtime, blueprint, reportHelpers,
            params.helper_python, 'anat_cortical_pair', params.threads.toString().toInteger(), params.device)
        def ch_anat_cortical_pair = ANAT_CORTICAL_PAIR.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_cortical_pair'): result])
        }

        def input_anat_pial_ribbon_qc = joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(ch_anat_pial_surface, ch_anat_ribbon), ch_anat_segmentation_qc), ch_anat_spherical_registration_qc), ch_anat_surface_init_qc), ch_anat_volume_support_qc), ch_anat_white_cortex_qc), ch_session_alignment_qc)
        ANAT_PIAL_RIBBON_QC(stageInput(joinLineages(input_anat_pial_ribbon_qc, ch_anat_cortical_pair)), runtime, blueprint, reportHelpers,
            params.helper_python, 'anat_pial_ribbon_qc', params.qc_mode, params.qc_surface_engine)
        def completed_anat_pial_ribbon_qc = ANAT_PIAL_RIBBON_QC.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_pial_ribbon_qc'): result])
        }
        def ch_anat_pial_ribbon_qc = params.qc_mode == 'strict' ? completed_anat_pial_ribbon_qc : input_anat_pial_ribbon_qc

        ANAT_APAS2ASEG(stageInput(joinLineages(ch_anat_pial_ribbon_qc, ch_anat_white_cortex_qc)), runtime, blueprint, helpers,
            params.helper_python, 'anat_apas2aseg', params.threads.toString().toInteger(), params.device)
        def ch_anat_apas2aseg = ANAT_APAS2ASEG.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_apas2aseg'): result])
        }

        ANAT_PARCSTATS(stageInput(joinLineages(joinLineages(joinLineages(ch_anat_pial_ribbon_qc, ch_anat_spherical_registration_qc), ch_anat_volume_support_qc), ch_anat_white_cortex_qc)), runtime, blueprint, helpers,
            params.helper_python, 'anat_parcstats', params.threads.toString().toInteger(), params.device)
        def ch_anat_parcstats = ANAT_PARCSTATS.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_parcstats'): result])
        }

        ANAT_PCTSURFCON(stageInput(joinLineages(joinLineages(joinLineages(ch_anat_motioncor, ch_anat_pial_ribbon_qc), ch_anat_spherical_registration_qc), ch_anat_white_cortex_qc)), runtime, blueprint, helpers,
            params.helper_python, 'anat_pctsurfcon', params.threads.toString().toInteger(), params.device)
        def ch_anat_pctsurfcon = ANAT_PCTSURFCON.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_pctsurfcon'): result])
        }

        VALIDATE_FSAVERAGE6_SPHERE_CORRESPONDENCE(stageInput(joinLineages(joinLineages(joinLineages(ch_anat_pial_ribbon_qc, ch_anat_smooth_inflated), ch_anat_spherical_registration_qc), ch_anat_white_cortex_qc)), runtime, blueprint, helpers,
            params.helper_python, 'validate_fsaverage6_sphere_correspondence', params.threads.toString().toInteger(), params.device)
        def ch_validate_fsaverage6_sphere_correspondence = VALIDATE_FSAVERAGE6_SPHERE_CORRESPONDENCE.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('validate_fsaverage6_sphere_correspondence'): result])
        }

        ANAT_APARC2ASEG(stageInput(joinLineages(joinLineages(joinLineages(ch_anat_apas2aseg, ch_anat_pial_ribbon_qc), ch_anat_spherical_registration_qc), ch_anat_white_cortex_qc)), runtime, blueprint, helpers,
            params.helper_python, 'anat_aparc2aseg', params.threads.toString().toInteger(), params.device)
        def ch_anat_aparc2aseg = ANAT_APARC2ASEG.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_aparc2aseg'): result])
        }

        RESAMPLE_WHITE_SURFACE_FSAVERAGE6(stageInput(joinLineages(joinLineages(joinLineages(joinLineages(ch_anat_motioncor, ch_anat_spherical_registration_qc), ch_anat_talairach_and_nu), ch_anat_white_cortex_qc), ch_validate_fsaverage6_sphere_correspondence)), runtime, blueprint, reportHelpers,
            params.helper_python, 'resample_white_surface_fsaverage6', params.threads.toString().toInteger(), params.device)
        def ch_resample_white_surface_fsaverage6 = RESAMPLE_WHITE_SURFACE_FSAVERAGE6.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('resample_white_surface_fsaverage6'): result])
        }

        def input_anat_native_outputs_qc = joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(ch_anat_aparc2aseg, ch_anat_apas2aseg), ch_anat_motioncor), ch_anat_parcstats), ch_anat_pctsurfcon), ch_anat_pial_ribbon_qc), ch_anat_spherical_registration_qc), ch_anat_volume_support_qc), ch_anat_white_cortex_qc)
        ANAT_NATIVE_OUTPUTS_QC(stageInput(input_anat_native_outputs_qc), runtime, blueprint, reportHelpers,
            params.helper_python, 'anat_native_outputs_qc', params.qc_mode, params.qc_surface_engine)
        def completed_anat_native_outputs_qc = ANAT_NATIVE_OUTPUTS_QC.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_native_outputs_qc'): result])
        }
        def ch_anat_native_outputs_qc = params.qc_mode == 'strict' ? completed_anat_native_outputs_qc : input_anat_native_outputs_qc

        RESAMPLE_PIAL_SURFACE_FSAVERAGE6(stageInput(joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(ch_anat_motioncor, ch_anat_pial_ribbon_qc), ch_anat_spherical_registration_qc), ch_anat_talairach_and_nu), ch_resample_white_surface_fsaverage6), ch_validate_fsaverage6_sphere_correspondence)), runtime, blueprint, reportHelpers,
            params.helper_python, 'resample_pial_surface_fsaverage6', params.threads.toString().toInteger(), params.device)
        def ch_resample_pial_surface_fsaverage6 = RESAMPLE_PIAL_SURFACE_FSAVERAGE6.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('resample_pial_surface_fsaverage6'): result])
        }

        def input_anat_fsaverage6_qc = joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(ch_anat_motioncor, ch_anat_pial_ribbon_qc), ch_anat_smooth_inflated), ch_anat_spherical_registration_qc), ch_anat_talairach_and_nu), ch_anat_white_cortex_qc), ch_resample_pial_surface_fsaverage6), ch_resample_white_surface_fsaverage6), ch_validate_fsaverage6_sphere_correspondence)
        ANAT_FSAVERAGE6_QC(stageInput(input_anat_fsaverage6_qc), runtime, blueprint, reportHelpers,
            params.helper_python, 'anat_fsaverage6_qc', params.qc_mode, params.qc_surface_engine)
        def completed_anat_fsaverage6_qc = ANAT_FSAVERAGE6_QC.out.result.map { subject, lineage, result ->
            tuple(subject, lineage + [('anat_fsaverage6_qc'): result])
        }
        def ch_anat_fsaverage6_qc = params.qc_mode == 'strict' ? completed_anat_fsaverage6_qc : input_anat_fsaverage6_qc

        def completed_qc = joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(joinLineages(completed_anat_input_qc, completed_session_alignment_qc), completed_anat_segmentation_qc), completed_anat_preproc_qc), completed_anat_volume_support_qc), completed_anat_surface_init_qc), completed_anat_white_cortex_qc), completed_anat_spherical_registration_qc), completed_anat_pial_ribbon_qc), completed_anat_native_outputs_qc), completed_anat_fsaverage6_qc)
        ASSEMBLE(stageInput(joinLineages(completed_qc, ch_session_normalize_mni)),
                 runtime, blueprint, reportHelpers, params.helper_python)
        def reports = ANAT_INPUT_QC.out.reports.mix(SESSION_ALIGNMENT_QC.out.reports, ANAT_SEGMENTATION_QC.out.reports, ANAT_PREPROC_QC.out.reports, ANAT_VOLUME_SUPPORT_QC.out.reports, ANAT_SURFACE_INIT_QC.out.reports, ANAT_WHITE_CORTEX_QC.out.reports, ANAT_SPHERICAL_REGISTRATION_QC.out.reports, ANAT_PIAL_RIBBON_QC.out.reports, ANAT_NATIVE_OUTPUTS_QC.out.reports, ANAT_FSAVERAGE6_QC.out.reports).collect()
        COLLECT_QC(reports, helpers, params.helper_python)
    }
}
