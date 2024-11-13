---
title: btc-vanity
draft: false
tags:
  - Bitcoin
  - BitcoinWallet
  - Cryptography
  - Rust
  - Vanity
  - BitcoinAddress
  - Projects
---
[[https://github.com/Emivvvvv/btc-vanity|btc-vanity]] is A Bitcoin vanity address generator written with the Rust programming language.
<br>
See the [[https://docs.rs/btc-vanity/latest/btc_vanity/|documentation]] or [[https://crates.io/crates/btc-vanity|crates.io page]].

### What is a vanity address?

A Vanity Address is **a special address that contains a specific pattern or word**. For example, the Vanity Address of a Bitcoin belonging to a user may contain a pattern or word such as "1BTC" or "1Emiv". Vanity Addresses are often used personally or for promotional purposes.

### What btc-vanity can do?

With btc-vanity, you can find a private key that has a compressed Bitcoin pay address that has a custom prefix, suffix, or a string at somewhere in the address.

You can easily run the btc-vanity terminal application locally or use it as a library to create your vanity keypair securely.

### btc-vanity implementation

```
src
|- lib.rs
|- main.rs
|- cli.rs
|- decoration.rs
|- error.rs
|- file.rs
|- flags.rs
|- keys_and_addresses.rs
|- vanity_addr_generator.rs
```

despite there are a lot of files, the real magic happens in `keys_and_addresses.rs` and `vanity_addr_generator.rs`, and others that serve CLI operations.

let's dive into some core implementations.

#### keys_and_addresses.rs

secp256k1 refers to the parameters of the elliptic curve used in Bitcoin's public-key cryptography. Currently, Bitcoin uses secp256k1. So we also need to use it. The main struct used in `keys_and_addresses.rs` is `KeysAndAdress` struct. Here we use `bitcoin::crypto::key` for storing keys and normal good old friend `String` for our compressed address that will have out vanity string.
```rust
pub struct KeysAndAddress {  
    private_key: PrivateKey,  
    public_key: PublicKey,  
    comp_address: String,  
}
```
To generate a random Key pair and their address, you need to initialize a `bitcoin::secp256k1::Secp256k1` to pass it generate method of KeysAndAddress, or just call `KeysAndAddress::generate_random_heavy()` if you're not going to use secp256k1 anywhere else. Here is how `genereate_random()` is implemented

```rust
pub fn generate_random(secp256k1: &Secp256k1<All>) -> Self {  
    let (secret_key, pk) = secp256k1.generate_keypair(&mut rand::thread_rng());  
    let private_key = PrivateKey::new(secret_key, Bitcoin);  
    let public_key = PublicKey::new(pk);  
  
    KeysAndAddress {  
        private_key,  
        public_key,  
        comp_address: Address::p2pkh(public_key, Bitcoin).to_string(),  
    }  
}
```
You can also within a certain range by using `KeysAndAddress::generate_within_range()`. Refer to docs for further information; I'll stop here to keep it simple.

#### vanity_addr_generator.rs

All of the other files are helpers of this file, this is the place where the real magic happens. The main struct used in `vanity_addr_generator.rs` is `VanityAddr` struct. Here is how `VanityAddr::generate()` implemented
```rust
/// Checks all given information's before passing to the vanity address finder function.  
/// Returns Result<KeysAndAddressString, VanityGeneratorError>  
/// Returns OK if a vanity address found successfully with keys_and_address::KeysAndAddress struct  
/// Returns Err if the string is longer than 4 chars and -d or --disable-fast-mode flags are not given.  
/// Returns Err if the string is not in base58 format.  
pub fn generate(  
    string: &str,  
    threads: u64,  
    case_sensitive: bool,  
    fast_mode: bool,  
    vanity_mode: VanityMode,  
) -> Result<KeysAndAddress, BtcVanityError> {  
    let secp256k1 = Secp256k1::new();  
  
    Self::validate_input(string, fast_mode)?;  
  
    if string.is_empty() {  
        return Ok(KeysAndAddress::generate_random(&secp256k1));  
    }  
  
    Ok(SearchEngines::find_vanity_address(  
        string,  
        threads,  
        case_sensitive,  
        vanity_mode,  
        secp256k1,  
    ))  
}
```

as it is quite self-explanatory, you pass some parameters, it checks it first, then searches your vanity string for you.
<br>
Another important struct in the file is `SearchEngines`. Real work is done by `SearchEngines` struct. `VanityAddr`'s main purpose is to provide a safe wrapper for finding a vanity address. Refer to docs or GitHub repository for complete implementation. However, I'll give the overall steps on how the magic happens.

1- create a `mpsc::channel()`
<br>
2- clone `sender` and the flags as many as requested.
<br>
3- now each thread will generate a randomly generated `KeysAndAdress` struct and check if it satisfies the flags
<br>
4- if a thread finds a `KeysAndAdress` that meets the needs, it sends it via the `sender`. When a thread sends a `KeysAndAdress`, the other threads are killed.
<br>
5- during this time receiver is trying to receive with `.try_recv()` in a loop, when a thread finds a keypair, this will catch it and return it.

### Calling btc-vanity as a library

let's find a vanity address
```rust
 use btc_vanity::vanity_addr_generator::{VanityAddr, VanityMode};  
  
 let vanity_address = VanityAddr::generate(  
             "Emiv", // the string that you want your vanity address to include.  
             16, // number of threads  
             false, // case sensitivity (false ex: eMiv, true ex: Emiv)  
             true, // fast mode flag (set false to use a string longer than 4 chars)  
             VanityMode::Anywhere, // vanity mode flag (prefix, suffix, anywhere available)  
             ).unwrap(); // this function returns a result type  
  
 println!("private_key (wif): {}\n\  
           public_key (compressed): {}\n\  
           address (compressed): {}\n\n",  
                 vanity_address.get_wif_private_key(),  
                 vanity_address.get_comp_public_key(),  
                 vanity_address.get_comp_address())
```
incase you need to look for certain range, we also got you covered. Here is an example to look for a certain private key range

```rust
fn main() {  
    // Define the minimum range for the private key in hexadecimal format.  
    let range_min = BigUint::from_str_radix("0000000000000000000000000000000000000000000000100000000000000000", 16).unwrap();  
  
    // Define the maximum range for the private key in hexadecimal format.  
    let range_max = BigUint::from_str_radix("00000000000000000000000000000000000000000000001FFFFFFFFFFFFFFFFF", 16).unwrap();  
  
    // Generate a vanity address with the desired pattern within the specified range.  
    let vanity_address = VanityAddr::generate_within_range(  
        "abc",          // The string that you want your vanity address to include (as a prefix in this case).  
        range_min,      // The minimum value of the private key range.  
        range_max,      // The maximum value of the private key range.  
        16,             // The number of threads to use for faster processing.  
        true,           // Case sensitivity flag: true means exact match, false means case-insensitive match.  
        true,           // Fast mode flag: enables fast mode, limiting the string length to 4 characters.  
        VanityMode::Prefix, // Where to match the string in the address (Prefix in this case).  
    ).unwrap();         // Unwrap the result to get the generated address or panic on error.  
  
    // Print the generated private key, public key, and address in their respective formats.  
    println!(  
        "private_key (wif): {}\n\  
        public_key (compressed): {}\n\  
        address (compressed): {}\n\n",  
        vanity_address.get_wif_private_key(),  
        vanity_address.get_comp_public_key(),  
        vanity_address.get_comp_address()  
    )  
}
```
### Examples of usage from terminal

install btc-vanity with cargo
```
cargo install btc-vanity
```
and then you can see possible usage with command
```
btc-vanity -h
```
here are some example screenshots

![example](example.png)
![examples-input-file](examples-input-file.png)

Thanks for reading till the end. I hope you like the project. If you would like to test btc-vanity, before using it please read the [[https://github.com/Emivvvvv/btc-vanity?tab=readme-ov-file#disclaimer|DISCLAIMER]] for your good.