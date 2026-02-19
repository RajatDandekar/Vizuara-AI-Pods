#!/usr/bin/env python3
"""
One-time migration script: Reorganize VIZflix from flat courses to Course → Pod hierarchy.

What it does:
1. Creates new content/courses/{courseSlug}/pods/{podSlug}/ directories
2. Moves existing course.json → pod.json (with added courseSlug + order fields)
3. Moves article.md and case_study.md into pod dirs
4. Creates course.json for each parent course
5. Creates new catalog.json with course-level entries
6. Copies public assets into nested structure (figures, notebooks, case-studies)

Old structure is preserved (not deleted) so you can verify before cleanup.
"""

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT_DIR = ROOT / "content" / "courses"
PUBLIC_DIR = ROOT / "public"

# ─── Course ↔ Pod Mapping ────────────────────────────────────────────

COURSES = [
    {
        "slug": "diffusion-models",
        "title": "Principles of Diffusion Models",
        "description": "From variational autoencoders to video generation — understanding diffusion models from first principles.",
        "difficulty": "intermediate",
        "tags": ["diffusion", "generative-models", "video-generation"],
        "pods": [
            {"slug": "vae-explained", "title": "Variational Autoencoders Explained", "order": 1, "status": "new"},
            {"slug": "intro-ddpm", "title": "Introduction to Diffusion Models (DDPM)", "order": 2, "status": "new"},
            {"slug": "energy-based-models", "title": "Energy-Based Models & Score Function", "order": 3, "status": "new"},
            {"slug": "denoising-score-matching", "title": "Denoising Score Matching", "order": 4, "status": "new"},
            {"slug": "noise-conditional-score-networks", "title": "Noise Conditional Score Networks", "order": 5, "status": "new"},
            {"slug": "diffusion-models-video-generation", "title": "From Still to Motion: How Diffusion Models Learned to Generate Videos", "order": 6, "status": "live"},
        ],
    },
    {
        "slug": "rl-from-scratch",
        "title": "RL From Scratch",
        "description": "From Q-learning to GRPO — build reinforcement learning from first principles.",
        "difficulty": "intermediate",
        "tags": ["reinforcement-learning", "reasoning", "rlhf"],
        "pods": [
            {"slug": "basics-of-rl", "title": "Basics of Reinforcement Learning", "order": 1, "status": "new"},
            {"slug": "value-functions-q-learning", "title": "Value Functions and Q-Learning", "order": 2, "status": "new"},
            {"slug": "dqn-atari-agents", "title": "Building DQN Atari Agents", "order": 3, "status": "new"},
            {"slug": "policy-gradient-methods", "title": "Policy Gradient Methods", "order": 4, "status": "new"},
            {"slug": "rlhf-theory-implementation", "title": "RLHF Theory and Implementation", "order": 5, "status": "new"},
            {"slug": "grpo", "title": "Group-relative Policy Optimization", "order": 6, "status": "new"},
            {"slug": "building-reasoning-model", "title": "Building a Reasoning Model from Scratch", "order": 7, "status": "new"},
        ],
    },
    {
        "slug": "ai-in-production",
        "title": "AI in Production",
        "description": "Deploy, monitor, and govern ML systems — from model serving to CI/CD for ML.",
        "difficulty": "intermediate",
        "tags": ["mlops", "deployment", "monitoring"],
        "pods": [
            {"slug": "model-deployment-serving", "title": "Model Deployment & Serving", "order": 1, "status": "new"},
            {"slug": "monitoring-observability", "title": "Monitoring & Observability", "order": 2, "status": "new"},
            {"slug": "mlops-cicd", "title": "MLOps & CI/CD for ML", "order": 3, "status": "new"},
            {"slug": "data-engineering-feature-stores", "title": "Data Engineering & Feature Stores", "order": 4, "status": "new"},
            {"slug": "model-governance-reliability", "title": "Model Governance & Reliability", "order": 5, "status": "new"},
        ],
    },
    {
        "slug": "vlms-from-scratch",
        "title": "VLMs from Scratch",
        "description": "Build vision-language models from first principles — from ViT to multimodal instruction tuning.",
        "difficulty": "intermediate",
        "tags": ["vision-language", "transformers", "multimodal"],
        "pods": [
            {"slug": "vision-transformers-from-scratch", "title": "Vision Transformers from Scratch", "order": 1, "status": "live"},
            {"slug": "multimodal-fusion", "title": "Multimodal Fusion Architectures", "order": 2, "status": "new"},
            {"slug": "contrastive-pretraining-clip", "title": "Contrastive Pretraining (CLIP-style)", "order": 3, "status": "new"},
            {"slug": "cross-attention-token-alignment", "title": "Cross-Attention & Token Alignment", "order": 4, "status": "new"},
            {"slug": "multimodal-instruction-tuning", "title": "Multimodal Instruction Tuning", "order": 5, "status": "new"},
        ],
    },
    {
        "slug": "build-llm",
        "title": "Build LLM from Scratch",
        "description": "From BERT to GPT — build large language models from first principles.",
        "difficulty": "intermediate",
        "tags": ["language-models", "transformers", "nlp"],
        "pods": [
            {"slug": "understanding-bert-from-scratch", "title": "Understanding BERT from Scratch", "order": 1, "status": "live"},
            {"slug": "self-attention-first-principles", "title": "Self-Attention from First Principles", "order": 2, "status": "new"},
            {"slug": "gpt-from-scratch", "title": "Building a GPT-style Model from Scratch", "order": 3, "status": "new"},
            {"slug": "training-pipeline-engineering", "title": "Training Pipeline Engineering", "order": 4, "status": "new"},
            {"slug": "inference-and-scaling", "title": "Inference & Scaling", "order": 5, "status": "new"},
        ],
    },
    {
        "slug": "build-diffusion-llm",
        "title": "Build Diffusion LLM from Scratch",
        "description": "What if language models could generate all tokens at once — like image diffusion, but for text?",
        "difficulty": "intermediate",
        "tags": ["diffusion", "language-models"],
        "pods": [
            {"slug": "diffusion-llms-from-scratch", "title": "Diffusion LLMs from Scratch", "order": 1, "status": "live"},
        ],
    },
    {
        "slug": "gpu-programming",
        "title": "GPU Programming from Scratch",
        "description": "From CUDA basics to distributed GPU systems — master GPU programming for deep learning.",
        "difficulty": "intermediate",
        "tags": ["gpu", "cuda", "distributed-training", "systems"],
        "pods": [
            {"slug": "5d-parallelism-from-scratch", "title": "5D Parallelism Foundations", "order": 1, "status": "live"},
            {"slug": "gpu-architecture-deep-dive", "title": "GPU Architecture Deep Dive", "order": 2, "status": "new"},
            {"slug": "cuda-programming-from-scratch", "title": "CUDA Programming Model from Scratch", "order": 3, "status": "new"},
            {"slug": "memory-hierarchy-optimization", "title": "Memory Hierarchy Optimization", "order": 4, "status": "new"},
            {"slug": "memory-coalescing-throughput", "title": "Memory Coalescing, Bank Conflicts & Throughput", "order": 5, "status": "new"},
            {"slug": "high-performance-cuda-kernels", "title": "Writing High-Performance CUDA Kernels", "order": 6, "status": "new"},
            {"slug": "kernel-fusion-optimization", "title": "Kernel Fusion & Operator Optimization", "order": 7, "status": "new"},
            {"slug": "mixed-precision-tensor-cores", "title": "Mixed Precision & Tensor Cores", "order": 8, "status": "new"},
            {"slug": "profiling-debugging", "title": "Profiling & Debugging", "order": 9, "status": "new"},
            {"slug": "distributed-gpu-systems", "title": "Distributed GPU Systems", "order": 10, "status": "new"},
        ],
    },
    {
        "slug": "modern-robot-learning",
        "title": "Modern Robot Learning",
        "description": "From world models to VLAs — build modern robot learning systems from scratch.",
        "difficulty": "intermediate",
        "tags": ["robotics", "world-models", "vla"],
        "pods": [
            {"slug": "act-policies", "title": "ACT (Action Chunking Transformer) Policies", "order": 1, "status": "new"},
            {"slug": "diffusion-policy-visuomotor", "title": "Diffusion Policy for Visuomotor Control", "order": 2, "status": "new"},
            {"slug": "vla-models", "title": "Vision-Language-Action (VLA) Models", "order": 3, "status": "new"},
            {"slug": "world-models", "title": "Understanding World Models from Scratch", "order": 4, "status": "live"},
            {"slug": "offline-rl-robot-learning", "title": "Offline RL & Dataset-Driven Robot Learning", "order": 5, "status": "new"},
            {"slug": "behavior-cloning-at-scale", "title": "Behavior Cloning at Scale (RT-X Style)", "order": 6, "status": "new"},
            {"slug": "foundation-models-embodied", "title": "Foundation Models for Embodied Intelligence", "order": 7, "status": "new"},
        ],
    },
    {
        "slug": "vlas-autonomous-driving",
        "title": "VLAs for Autonomous Driving",
        "description": "How combining vision, language understanding, and action generation is reshaping autonomous driving.",
        "difficulty": "advanced",
        "tags": ["autonomous-driving", "vla", "robotics"],
        "pods": [
            {"slug": "vla-autonomous-driving", "title": "VLA Models for Autonomous Driving", "order": 1, "status": "live"},
        ],
    },
    {
        "slug": "tiny-recursive-models",
        "title": "Tiny Recursive Models",
        "description": "How a 7-million parameter network outperforms trillion-parameter LLMs on reasoning benchmarks.",
        "difficulty": "beginner",
        "tags": ["reasoning", "efficiency", "recursive-models"],
        "pods": [
            {"slug": "tiny-recursive-models", "title": "Tiny Recursive Models: Less is More for AI Reasoning", "order": 1, "status": "live"},
        ],
    },
    {
        "slug": "context-engineering",
        "title": "Context Engineering from Scratch",
        "description": "The art and science of filling the context window with the right information.",
        "difficulty": "intermediate",
        "tags": ["context-engineering", "rag", "prompt-design"],
        "pods": [
            {"slug": "context-engineering-for-llms", "title": "Context Engineering for LLMs", "order": 1, "status": "live"},
            {"slug": "prompt-design-principles", "title": "Prompt Design Principles", "order": 2, "status": "new"},
            {"slug": "rag-systems", "title": "Retrieval-Augmented Generation (RAG) Systems", "order": 3, "status": "new"},
            {"slug": "memory-architectures", "title": "Memory Architectures", "order": 4, "status": "new"},
            {"slug": "context-optimization-evaluation", "title": "Context Optimization & Evaluation", "order": 5, "status": "new"},
        ],
    },
    {
        "slug": "voice-ai-agents",
        "title": "Voice AI Agents from Scratch",
        "description": "Build real-time voice AI agents with speech pipelines and conversational state management.",
        "difficulty": "intermediate",
        "tags": ["voice-ai", "agents", "speech"],
        "pods": [
            {"slug": "real-time-speech-pipeline", "title": "Real-Time Speech Pipeline", "order": 1, "status": "new"},
            {"slug": "conversational-voice-agents", "title": "Conversational State & Tool-Integrated Voice Agents", "order": 2, "status": "new"},
        ],
    },
    {
        "slug": "llm-evaluation",
        "title": "LLM Evaluation",
        "description": "Benchmark, evaluate, and audit large language models — from reasoning to safety.",
        "difficulty": "intermediate",
        "tags": ["evaluation", "benchmarking", "safety"],
        "pods": [
            {"slug": "benchmarking-fundamentals", "title": "Benchmarking Fundamentals", "order": 1, "status": "new"},
            {"slug": "reasoning-cot-evaluation", "title": "Reasoning & Chain-of-Thought Evaluation", "order": 2, "status": "new"},
            {"slug": "safety-bias-alignment", "title": "Safety, Bias & Alignment Audits", "order": 3, "status": "new"},
            {"slug": "human-eval-llm-judge", "title": "Human Evaluation & LLM-as-a-Judge", "order": 4, "status": "new"},
        ],
    },
]


def migrate_live_pod(pod_slug: str, course_slug: str, order: int):
    """Migrate an existing live pod into the new hierarchy."""
    old_content_dir = CONTENT_DIR / pod_slug
    new_pod_dir = CONTENT_DIR / course_slug / "pods" / pod_slug

    if not old_content_dir.exists():
        print(f"  SKIP (no content dir): {pod_slug}")
        return None

    new_pod_dir.mkdir(parents=True, exist_ok=True)

    # Read old course.json → transform to pod.json
    old_manifest_path = old_content_dir / "course.json"
    if old_manifest_path.exists():
        manifest = json.loads(old_manifest_path.read_text())
        manifest["courseSlug"] = course_slug
        manifest["order"] = order
        (new_pod_dir / "pod.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  Created pod.json for {pod_slug}")
    else:
        print(f"  WARNING: No course.json for {pod_slug}")
        return None

    # Copy article.md
    old_article = old_content_dir / "article.md"
    if old_article.exists():
        shutil.copy2(old_article, new_pod_dir / "article.md")
        print(f"  Copied article.md")

    # Copy case_study.md
    old_case_study = old_content_dir / "case_study.md"
    if old_case_study.exists():
        shutil.copy2(old_case_study, new_pod_dir / "case_study.md")
        print(f"  Copied case_study.md")

    # Copy public/courses/{slug}/figures → public/courses/{course}/pods/{pod}/figures
    old_figures = PUBLIC_DIR / "courses" / pod_slug / "figures"
    new_figures = PUBLIC_DIR / "courses" / course_slug / "pods" / pod_slug / "figures"
    if old_figures.exists():
        if new_figures.exists():
            shutil.rmtree(new_figures)
        shutil.copytree(old_figures, new_figures)
        fig_count = len([f for f in new_figures.iterdir() if f.is_file()])
        print(f"  Copied {fig_count} figure(s)")

    # Copy public/notebooks/{slug}/ → public/notebooks/{course}/{pod}/
    old_notebooks = PUBLIC_DIR / "notebooks" / pod_slug
    new_notebooks = PUBLIC_DIR / "notebooks" / course_slug / pod_slug
    if old_notebooks.exists():
        if new_notebooks.exists():
            shutil.rmtree(new_notebooks)
        shutil.copytree(old_notebooks, new_notebooks)
        print(f"  Copied notebooks directory")

    # Copy public/case-studies/{slug}/ → public/case-studies/{course}/{pod}/
    old_cases = PUBLIC_DIR / "case-studies" / pod_slug
    new_cases = PUBLIC_DIR / "case-studies" / course_slug / pod_slug
    if old_cases.exists():
        if new_cases.exists():
            shutil.rmtree(new_cases)
        shutil.copytree(old_cases, new_cases)
        print(f"  Copied case study assets")

    # Return pod card data
    return {
        "slug": pod_slug,
        "title": manifest.get("title", pod_slug),
        "description": manifest.get("description", ""),
        "order": order,
        "notebookCount": len(manifest.get("notebooks", [])),
        "estimatedHours": manifest.get("estimatedHours", 1),
        "hasCaseStudy": "caseStudy" in manifest and manifest["caseStudy"] is not None,
        "thumbnail": f"/courses/{course_slug}/pods/{pod_slug}/figures/figure_1.png",
    }


def create_new_pod_placeholder(pod_slug: str, pod_title: str, course_slug: str, order: int):
    """Create a placeholder directory for a new (not yet created) pod."""
    new_pod_dir = CONTENT_DIR / course_slug / "pods" / pod_slug
    new_pod_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal pod.json
    pod_json = {
        "title": pod_title,
        "slug": pod_slug,
        "courseSlug": course_slug,
        "order": order,
        "description": f"Coming soon: {pod_title}",
        "difficulty": "intermediate",
        "estimatedHours": 1,
        "prerequisites": [],
        "tags": [],
        "article": {"figureUrls": {}},
        "notebooks": [],
    }
    (new_pod_dir / "pod.json").write_text(json.dumps(pod_json, indent=2) + "\n")

    return {
        "slug": pod_slug,
        "title": pod_title,
        "description": f"Coming soon: {pod_title}",
        "order": order,
        "notebookCount": 0,
        "estimatedHours": 1,
        "hasCaseStudy": False,
    }


def main():
    print("=" * 60)
    print("VIZflix Migration: Flat Courses → Course → Pod Hierarchy")
    print("=" * 60)

    catalog_courses = []

    for course_def in COURSES:
        course_slug = course_def["slug"]
        print(f"\n{'─' * 50}")
        print(f"Course: {course_def['title']} ({course_slug})")
        print(f"{'─' * 50}")

        course_dir = CONTENT_DIR / course_slug
        course_dir.mkdir(parents=True, exist_ok=True)
        (course_dir / "pods").mkdir(exist_ok=True)

        pod_cards = []
        total_notebooks = 0
        total_hours = 0
        live_pod_count = 0
        first_live_thumbnail = None

        for pod_def in course_def["pods"]:
            pod_slug = pod_def["slug"]
            pod_order = pod_def["order"]
            pod_status = pod_def["status"]

            print(f"\n  Pod {pod_order}: {pod_def['title']} [{pod_status}]")

            if pod_status == "live":
                pod_card = migrate_live_pod(pod_slug, course_slug, pod_order)
                if pod_card:
                    pod_cards.append(pod_card)
                    total_notebooks += pod_card["notebookCount"]
                    total_hours += pod_card["estimatedHours"]
                    live_pod_count += 1
                    if not first_live_thumbnail and pod_card.get("thumbnail"):
                        first_live_thumbnail = pod_card["thumbnail"]
            else:
                pod_card = create_new_pod_placeholder(pod_slug, pod_def["title"], course_slug, pod_order)
                pod_cards.append(pod_card)

        # Create course.json
        course_manifest = {
            "title": course_def["title"],
            "slug": course_slug,
            "description": course_def["description"],
            "difficulty": course_def["difficulty"],
            "estimatedHours": total_hours if total_hours > 0 else len(course_def["pods"]),
            "tags": course_def["tags"],
            "pods": pod_cards,
        }

        if first_live_thumbnail:
            course_manifest["thumbnail"] = first_live_thumbnail

        (course_dir / "course.json").write_text(json.dumps(course_manifest, indent=2) + "\n")
        print(f"\n  Created course.json ({len(pod_cards)} pods, {total_notebooks} notebooks)")

        # Add to catalog
        # Determine course status: live if at least one pod is live
        course_status = "live" if live_pod_count > 0 else "draft"

        catalog_entry = {
            "slug": course_slug,
            "title": course_def["title"],
            "description": course_def["description"],
            "difficulty": course_def["difficulty"],
            "estimatedHours": total_hours if total_hours > 0 else len(course_def["pods"]),
            "tags": course_def["tags"],
            "podCount": len(pod_cards),
            "totalNotebooks": total_notebooks,
            "status": course_status,
        }

        if first_live_thumbnail:
            catalog_entry["thumbnail"] = first_live_thumbnail

        catalog_courses.append(catalog_entry)

    # Write new catalog.json
    new_catalog = {"courses": catalog_courses}
    catalog_path = CONTENT_DIR / "catalog.json"

    # Backup old catalog
    old_catalog_path = CONTENT_DIR / "catalog.json.bak"
    if catalog_path.exists():
        shutil.copy2(catalog_path, old_catalog_path)
        print(f"\nBacked up old catalog to {old_catalog_path}")

    catalog_path.write_text(json.dumps(new_catalog, indent=2) + "\n")
    print(f"\nWrote new catalog.json ({len(catalog_courses)} courses)")

    # Summary
    print(f"\n{'=' * 60}")
    print("Migration complete!")
    print(f"{'=' * 60}")
    total_pods = sum(len(c["pods"]) for c in COURSES)
    live_pods = sum(1 for c in COURSES for p in c["pods"] if p["status"] == "live")
    print(f"  Courses: {len(COURSES)}")
    print(f"  Total pods: {total_pods} ({live_pods} migrated from live, {total_pods - live_pods} new placeholders)")
    print(f"\nNote: Old content directories are preserved. Delete them after verification.")


if __name__ == "__main__":
    main()
