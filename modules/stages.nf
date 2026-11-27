def quote(value) {
    return "'" + value.toString().replace("'", "'\"'\"'") + "'"
}

process PREFLIGHT {
    tag 'preflight'
    cpus 1
    input:
    path runtime
    path blueprint
    path helpers, stageAs: 'helpers'
    val python
    output:
    path 'checked/runtime.json', emit: runtime
    path 'checked/workflow.json', emit: blueprint
    path 'checked/subjects.txt', emit: subjects
    script:
    """
    ${quote(python)} -B helpers/preflight.py --runtime ${quote(runtime)} --blueprint ${quote(blueprint)}
    """
    stub:
    """
    ${quote(python)} -B helpers/preflight.py --stub --runtime ${quote(runtime)} --blueprint ${quote(blueprint)}
    """
}

process RUN_NODE {
    errorStrategy { params.isolate_session_errors ? 'ignore' : 'terminate' }
    tag "${subject}:${node_id}"
    cpus { threads }
    input:
    tuple val(subject), val(lineage), path(upstream_results, stageAs: 'upstream??/*', arity: '0..*')
    path runtime
    path blueprint
    path helpers, stageAs: 'helpers'
    val python
    val node_id
    val threads
    val device
    output:
    tuple val(subject), val(lineage), path('result'), emit: result
    script:
    def settings = [node_id: node_id, subject: subject, threads: task.cpus,
        device: device, upstream: upstream_results.collect { directory -> directory.toString() }]
    if (node_id == 'session_align_companions' && task.ext.companion_workers != null) {
        settings.companion_workers = task.ext.companion_workers
    }
    def spec = groovy.json.JsonOutput.toJson(settings)
    """
    cat > spec.json <<'SPEC'
    ${spec}
    SPEC
    ${quote(python)} -B helpers/run_node.py --spec spec.json --runtime ${quote(runtime)} --blueprint ${quote(blueprint)}
    """
    stub:
    def spec = groovy.json.JsonOutput.toJson([node_id: node_id, subject: subject, threads: task.cpus,
        upstream: upstream_results.collect { directory -> directory.toString() }])
    """
    cat > spec.json <<'SPEC'
    ${spec}
    SPEC
    ${quote(python)} -B helpers/stub_node.py --spec spec.json
    """
}

def surfaceSpec(nodeId, subject, cpus, device, backends, bundle, upstreamResults) {
    def assets = nodeId == 'anat_fix_topology' ? [topology_runtime: bundle.toString()] : [pial_runtime: bundle.toString()]
    return groovy.json.JsonOutput.toJson(backends + assets + [node_id: nodeId, subject: subject, threads: cpus,
        device: device, upstream: upstreamResults.collect { it.toString() }])
}

process RUN_SURFACE {
    errorStrategy { params.isolate_session_errors ? 'ignore' : 'terminate' }
    tag "${subject}:${node_id}"
    cpus { threads }
    input:
    tuple val(subject), val(lineage), path(upstream_results, stageAs: 'upstream??/*', arity: '0..*')
    path runtime
    path blueprint
    path helpers, stageAs: 'helpers'
    val python
    val node_id
    val threads
    val device
    val backends
    path bundle, stageAs: 'surface-runtime'
    output:
    tuple val(subject), val(lineage), path('result'), emit: result
    script:
    def spec = surfaceSpec(node_id, subject, task.cpus, device, backends, bundle, upstream_results)
    """
    cat > spec.json <<'SPEC'
    ${spec}
    SPEC
    ${quote(python)} -B helpers/run_node.py --spec spec.json --runtime ${quote(runtime)} --blueprint ${quote(blueprint)}
    """
    stub:
    def spec = surfaceSpec(node_id, subject, task.cpus, device, backends, bundle, upstream_results)
    """
    cat > spec.json <<'SPEC'
    ${spec}
    SPEC
    ${quote(python)} -B helpers/stub_node.py --spec spec.json
    """
}

process RUN_QC {
    errorStrategy { params.isolate_session_errors ? 'ignore' : 'terminate' }
    tag "${subject}:${node_id}"
    cpus { params.qc_threads }
    input:
    tuple val(subject), val(lineage), path(upstream_results, stageAs: 'upstream??/*', arity: '0..*')
    path runtime
    path blueprint
    path helpers, stageAs: 'helpers'
    val python
    val node_id
    val mode
    val surface_engine
    output:
    tuple val(subject), val(lineage), path('result'), emit: result
    path 'qc_result', emit: reports
    script:
    def spec = groovy.json.JsonOutput.toJson([node_id: node_id, subject: subject, threads: task.cpus,
        upstream: upstream_results.collect { directory -> directory.toString() }])
    """
    cat > spec.json <<'SPEC'
    ${spec}
    SPEC
    ${quote(python)} -B helpers/qc.py --spec spec.json --runtime ${quote(runtime)} --blueprint ${quote(blueprint)} --mode ${quote(mode)} --volume-engine batched --surface-engine ${quote(surface_engine)}
    """
    stub:
    def spec = groovy.json.JsonOutput.toJson([node_id: node_id, subject: subject,
        upstream: upstream_results.collect { directory -> directory.toString() }])
    """
    cat > spec.json <<'SPEC'
    ${spec}
    SPEC
    ${quote(python)} -B helpers/stub_node.py --spec spec.json
    mkdir -p qc_result
    cp spec.json qc_result/topology.json
    """
}

process ASSEMBLE {
    errorStrategy { params.isolate_session_errors ? 'ignore' : 'terminate' }
    tag "$subject"
    cpus 1
    publishDir params.outdir, mode: 'copy', overwrite: params.isolate_session_errors,
        saveAs: { filename -> filename.toString().replaceFirst('^published/', '') }
    input:
    tuple val(subject), val(lineage), path(upstream_results, stageAs: 'upstream??/*', arity: '0..*')
    path runtime
    path blueprint
    path helpers, stageAs: 'helpers'
    val python
    output:
    tuple val(subject), path('published/*'), emit: subjects
    script:
    def spec = groovy.json.JsonOutput.toJson([subject: subject,
        upstream: upstream_results.collect { directory -> directory.toString() }])
    """
    cat > spec.json <<'SPEC'
    ${spec}
    SPEC
    ${quote(python)} -B helpers/assemble.py --spec spec.json --runtime ${quote(runtime)} --blueprint ${quote(blueprint)}
    """
    stub:
    def spec = groovy.json.JsonOutput.toJson([subject: subject,
        upstream: upstream_results.collect { directory -> directory.toString() }])
    """
    cat > spec.json <<'SPEC'
    ${spec}
    SPEC
    ${quote(python)} -B helpers/assemble.py --stub --spec spec.json --runtime ${quote(runtime)} --blueprint ${quote(blueprint)}
    """
}

process COLLECT_QC {
    tag 'qc-index'
    cpus 1
    publishDir "${params.outdir}/qc", mode: 'copy', overwrite: params.isolate_session_errors,
        saveAs: { filename -> filename.toString().replaceFirst('^qc_index/', '') }
    input:
    path directories, stageAs: 'qc??/*', arity: '1..*'
    path helpers, stageAs: 'helpers'
    val python
    output:
    path 'qc_index/*', emit: index
    script:
    def arguments = directories.collect { directory -> quote(directory) }.join(' ')
    """
    ${quote(python)} -B helpers/collect_qc.py --directories ${arguments}
    """
    stub:
    """
    mkdir -p qc_index
    printf '%s\\n' '{"mode":"topology_only","scientific_approval":"not_performed"}' > qc_index/summary.json
    """
}
