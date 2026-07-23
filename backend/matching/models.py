"""Plain Django ORM models - no raw SQL or MySQL-specific features, so
switching config/settings.py's DATABASES to MySQL later needs no model
changes (see that file's DATABASES comment).

"Users" is deliberately not a custom model here - django.contrib.auth.User
already covers it and is already auth-ready for later.
"""
from django.conf import settings
from django.db import models


class SubmittedResume(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="submitted_resumes",
    )
    resume_text = models.TextField()
    original_filename = models.CharField(max_length=255, null=True, blank=True)
    submitted_at = models.DateTimeField(auto_now_add=True)
    # Keyword-overlap percentage and missing-keyword list between this resume
    # and its rank-1 match's JD text - see matching/services.py's compute_ats_analysis().
    ats_score = models.FloatField(null=True, blank=True)
    skill_gap = models.JSONField(default=list, blank=True)

    def __str__(self):
        return f"SubmittedResume #{self.pk} ({self.submitted_at:%Y-%m-%d %H:%M})"


class MatchResult(models.Model):
    submitted_resume = models.ForeignKey(
        SubmittedResume, on_delete=models.CASCADE, related_name="matches"
    )
    rank = models.PositiveSmallIntegerField()
    job_doc_id = models.CharField(max_length=64)
    category = models.CharField(max_length=255)
    title = models.CharField(max_length=500)
    score = models.FloatField()
    # [{"resume_chunk": str, "jd_chunk": str, "score": float}, ...] - see
    # matching/services.py's get_matches() / ml/src/explainability/schema.py's
    # ChunkMatch, which this is a plain-dict serialization of.
    explanations = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["rank"]

    def __str__(self):
        return f"#{self.rank} {self.job_doc_id} ({self.score:.3f}) for resume #{self.submitted_resume_id}"
