# Serialization

Kubetorch serializes call arguments, return values, and remote exceptions across
an HTTP boundary.

## Prefer JSON-compatible values

Strings, numbers, booleans, lists, dictionaries, and small structured records
are the most portable choices. They are easy to inspect in logs and safe to
consume from different client versions.

## Enable pickle only for trusted code

Pickle supports richer Python objects but can execute code during loading. Only
enable it when both endpoints and all payloads are trusted.

```python
compute = kt.Compute(
    cpus="2",
    allowed_serialization=["json", "pickle"],
)
```

## Move large data out of the response

Do not return checkpoints or large tensor batches through the callable response.
Write them to a volume, the Kubetorch data store, or object storage and return a
key plus metadata. Batch runs can register that key as an artifact URI.

## Version your contracts

If a remote class returns application-specific dictionaries, give them an
explicit schema/version field. That makes warm services safer when the local
driver changes during rapid iteration.
